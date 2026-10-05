"""Real-weight ASTE runs with validation-only checkpoint selection."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import random
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hoasa import HoasaRecord, load_hoasa, write_pyabsa_explicit_view
from .metrics import evaluate_triplets
from .pyabsa_compat import (
    IndonesianStanzaParser,
    build_pyabsa_training_classes,
    download_indonesian_parser,
    install_memory_efficient_graph_layers,
    predict_triplets,
    read_token_maps,
)
from .source_data import audit_sources


ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "configs" / "experiment-matrix.json"
DATA_ROOT = ROOT / "data" / "sources"
RUNS_ROOT = ROOT / "runs"


class RunBlocked(RuntimeError):
    """A declared experiment cell cannot start without its human/data gate."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_state() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}
    return {"revision": revision, "dirty": bool(status)}


def _module_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("pyabsa", "torch", "transformers", "stanza", "huggingface-hub"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _seed_runtime(torch: Any, seed: int) -> dict[str, Any]:
    import numpy as np

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "python_random": seed,
        "numpy": seed,
        "torch": seed,
        "torch_cuda": seed if torch.cuda.is_available() else None,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def _resolve_encoder(encoder: str, matrix: dict[str, Any]) -> tuple[Path, dict[str, str]]:
    spec = matrix["encoders"][encoder]
    revision = spec.get("model_revision")
    if not revision or len(revision) != 40:
        raise RunBlocked(f"encoder {encoder} has no pinned 40-character model revision")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RunBlocked("huggingface_hub is unavailable in the pinned runtime") from error
    snapshot = Path(
        snapshot_download(
            repo_id=spec["model_id"],
            revision=revision,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.model",
                "*.safetensors",
                "pytorch_model*.bin",
                "spm.model",
                "merges.txt",
                "vocab.json",
                "vocab.txt",
                "tokenizer*",
                "special_tokens_map.json",
                "added_tokens.json",
            ],
        )
    )
    if not snapshot.is_dir():
        raise RunBlocked(f"could not resolve pinned encoder snapshot for {encoder}")
    tokenizer_hashes = {}
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        if path.suffix.lower() not in {".json", ".txt", ".model", ".vocab"}:
            continue
        if path.stat().st_size > 16 * 1024 * 1024:
            continue
        tokenizer_hashes[path.relative_to(snapshot).as_posix()] = _sha256(path)
    return snapshot, {
        "repository": spec["model_id"],
        "revision": revision,
        "tokenizer_and_config_file_sha256": tokenizer_hashes,
    }


def _copy_records_for_smoke(records: list[HoasaRecord], limit: int | None) -> list[HoasaRecord]:
    if limit is None:
        return records
    # Source IDs remain stable for a prefix smoke subset and are revalidated by
    # the adapter before any run is allowed to use these rows.
    return records[:limit]


def _resolve_max_seq_len(
    *, encoder: str, matrix: dict[str, Any], source_hashes: dict[str, str]
) -> tuple[int, dict[str, Any]]:
    """Validate and apply the pre-audited profile for pinned data/tokenizer revisions."""
    profile = matrix["training"].get("max_seq_len_profiles", {}).get(encoder)
    if not isinstance(profile, dict):
        raise RunBlocked(f"no audited full-sequence length profile exists for {encoder}")
    expected_source_hashes = matrix["training"].get("max_seq_len_profile_source_hashes")
    if expected_source_hashes != source_hashes:
        raise RunBlocked("train/dev source hashes differ from the sequence-length audit")
    if profile.get("encoder_revision") != matrix["encoders"][encoder].get("model_revision"):
        raise RunBlocked("encoder revision differs from the sequence-length audit")
    minimum = int(matrix["training"]["max_seq_len"])
    resolved = int(profile["resolved_max_seq_len"])
    observed = max(
        int(profile["train_max_full_sequence_tokens"]),
        int(profile["dev_max_full_sequence_tokens"]),
    )
    if resolved < max(minimum, observed):
        raise RunBlocked("audited max sequence length would truncate a train/dev sentence")
    return resolved, {
        **profile,
        "minimum_from_plan": minimum,
        "resolved_max_seq_len": resolved,
        "full_sequence_tokens_max": observed,
        "splits_used_before_selection": ["train", "dev"],
        "source_hashes": source_hashes,
        "policy": matrix["training"]["max_seq_len_policy"],
    }


def _prepare_training_data(
    run_dir: Path,
    split_records: dict[str, list[HoasaRecord]],
) -> tuple[Path, dict[str, dict[int, list[int]]]]:
    prepared_dir = run_dir / "prepared"
    stage_dir = run_dir / "pyabsa-data"
    stage_dir.mkdir(parents=True, exist_ok=False)
    token_maps: dict[str, dict[int, list[int]]] = {}
    for split in ("train", "dev"):
        explicit_path = prepared_dir / f"{split}.explicit.txt"
        map_path = prepared_dir / f"{split}.token-map.jsonl"
        write_pyabsa_explicit_view(split_records[split], explicit_path, map_path)
        token_maps[split] = read_token_maps(map_path)
        shutil.copyfile(explicit_path, stage_dir / ("valid.txt" if split == "dev" else "train.txt"))
    # PyABSA's test slot is deliberately validation-only. The real held-out
    # test split is not staged or opened until a checkpoint has been selected.
    shutil.copyfile(stage_dir / "valid.txt", stage_dir / "test.txt")
    return stage_dir, token_maps


def _model_from_checkpoint(trainer: Any, checkpoint_dir: Path):
    import torch

    config = trainer.config
    model = config.model(config=config).to(config.device)
    install_memory_efficient_graph_layers(model)
    state_files = sorted(checkpoint_dir.glob("*.state_dict"))
    if len(state_files) != 1:
        raise RuntimeError("selected PyABSA checkpoint must contain one state dictionary")
    load_kwargs: dict[str, Any] = {"map_location": config.device}
    if "weights_only" in __import__("inspect").signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    state_dict = torch.load(state_files[0], **load_kwargs)
    model.load_state_dict(state_dict, strict=True)
    return model


def _external_test_evaluation(
    *,
    trainer: Any,
    dataset_class: Any,
    records: list[HoasaRecord],
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from pyabsa.framework.tokenizer_class.tokenizer_class import PretrainedTokenizer
    from pyabsa.tasks.AspectSentimentTripletExtraction.dataset_utils.aste_utils import DataIterator

    config = trainer.config
    test_records = records
    prepared_path = run_dir / "heldout-test.external.txt"
    map_path = run_dir / "heldout-test.token-map.jsonl"
    write_pyabsa_explicit_view(test_records, prepared_path, map_path)
    token_maps = read_token_maps(map_path)

    old_test_file = config.dataset_file["test"]
    config.dataset_file["test"] = str(prepared_path)
    try:
        test_set = dataset_class(
            config,
            PretrainedTokenizer(config),
            dataset_type="test",
        )
        test_set.convert_examples_to_features()
    finally:
        config.dataset_file["test"] = old_test_file

    test_loader = DataIterator(test_set, config=config)
    checkpoint_dir = Path(trainer.inference_model)
    model = _model_from_checkpoint(trainer, checkpoint_dir)
    predicted, decode_diagnostics = predict_triplets(
        model,
        test_loader,
        config,
        test_records,
        token_maps,
    )
    gold = [record.triplets for record in sorted(test_records, key=lambda item: item.row_id)]
    metrics = evaluate_triplets(gold, predicted)
    explicit_gold = [
        [triplet for triplet in record.triplets if triplet[0] != (-1,)]
        for record in sorted(test_records, key=lambda item: item.row_id)
    ]
    explicit_metrics = evaluate_triplets(explicit_gold, predicted)
    predictions_path = run_dir / "test-predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record, row_predictions in zip(sorted(test_records, key=lambda item: item.row_id), predicted):
            stream.write(
                json.dumps(
                    {
                        "row_id": record.row_id,
                        "predicted_triplets": [
                            [list(aspect), list(opinion), sentiment]
                            for aspect, opinion, sentiment in row_predictions
                        ],
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    return {
        "test": metrics,
        "test_explicit_aspects_only": explicit_metrics,
        "decode_diagnostics": decode_diagnostics,
        "test_evaluations": 1,
    }, {"predictions": predictions_path, "checkpoint_dir": checkpoint_dir}


def _resolved_optimizer_groups(observer: dict[str, Any]) -> list[dict[str, Any]]:
    groups = observer.get("optimizer_groups", [])
    return groups


def run_experiment(
    *,
    dataset: str,
    architecture: str,
    encoder: str,
    seed: int,
    purpose: str = "benchmark",
    smoke_examples: int = 24,
    download_parser: bool = False,
) -> dict[str, Any]:
    if purpose not in {"smoke", "benchmark"}:
        raise ValueError("purpose must be `smoke` or `benchmark`")
    if dataset != "hoasa":
        readiness = audit_sources()
        if dataset == "casa" and not readiness["casa"]["aste_gold_ready"]:
            blockers = "; ".join(readiness["casa"]["readiness_blockers"])
            raise RunBlocked(f"CASA ASTE gold is not ready: {blockers}")
        raise RunBlocked("only the HoASA in-domain path is implemented in this runner")
    if architecture not in {"emcgcn", "ssgcn"}:
        raise ValueError(f"undeclared architecture: {architecture}")

    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    if encoder not in matrix["encoders"]:
        raise ValueError(f"undeclared encoder: {encoder}")
    if seed not in [int(value) for value in matrix["seed_protocol"]["seeds"]]:
        raise ValueError("seed must be in the predeclared seed protocol")

    audit = audit_sources()
    canonical_paths = {
        split: DATA_ROOT / "hoasa" / f"{split}.txt" for split in ("train", "dev", "test")
    }
    # Read train and validation only before checkpoint selection. The pinned
    # source audit checks test integrity without passing its labels to training.
    train_records = load_hoasa(canonical_paths["train"])
    dev_records = load_hoasa(canonical_paths["dev"])
    if purpose == "smoke":
        limit = max(1, int(smoke_examples))
        train_records = _copy_records_for_smoke(train_records, limit)
        dev_records = _copy_records_for_smoke(dev_records, limit)
        test_row_count = min(limit, int(audit["hoasa"]["splits"]["test"]["sentences"]))
    else:
        test_row_count = int(audit["hoasa"]["splits"]["test"]["sentences"])

    run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-hoasa-"
        f"{architecture}-{encoder}-s{seed}-{purpose}-{uuid.uuid4().hex[:8]}"
    )
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment_protocol_id": matrix["experiment_protocol_id"],
        "run_id": run_id,
        "status": "running",
        "purpose": purpose,
        "dataset": dataset,
        "train_dataset": dataset,
        "test_dataset": dataset,
        "architecture": architecture,
        "encoder": encoder,
        "seed": int(seed),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_audit": audit,
        "source_files": {
            split: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for split, path in canonical_paths.items()
        },
        "source_rows": {
            "train": len(train_records),
            "dev": len(dev_records),
            "test": test_row_count,
        },
        "test_protocol": {
            "pyabsa_internal_test_alias": "dev",
            "heldout_test_evaluations": 1,
        },
        "code": _git_state(),
        "code_file_hashes": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in sorted((ROOT / "src" / "aste_repro").glob("*.py"))
        },
    }
    _json_write(manifest_path, manifest)
    log_path = run_dir / "training.log"

    try:
        import torch
        from pyabsa.tasks.AspectSentimentTripletExtraction.configuration.configuration import (
            ASTEConfigManager,
        )
        from pyabsa.tasks.AspectSentimentTripletExtraction.models.model import EMCGCN
        from pyabsa.tasks.AspectSentimentTripletExtraction.trainer.trainer import ASTETrainer
        from pyabsa.framework.dataset_class.dataset_dict_class import DatasetDict
        from pyabsa.framework.flag_class.flag_template import DeviceTypeOption, ModelSaveOption

        try:
            pyabsa_version = importlib.metadata.version("pyabsa")
        except importlib.metadata.PackageNotFoundError as error:
            raise RunBlocked("PyABSA is not installed in this Python environment") from error
        if pyabsa_version != matrix["pyabsa_compatibility"]["version"]:
            raise RunBlocked(f"expected PyABSA {matrix['pyabsa_compatibility']['version']}, found {pyabsa_version}")

        deterministic = _seed_runtime(torch, int(seed))
        resolved_model_dir, model_ref = _resolve_encoder(encoder, matrix)
        config = ASTEConfigManager.get_aste_config_base()
        config.pretrained_bert = str(resolved_model_dir)
        sequence_length_source_hashes = {
            split: _sha256(canonical_paths[split]) for split in ("train", "dev")
        }
        resolved_seq_len, seq_len_audit = _resolve_max_seq_len(
            encoder=encoder, matrix=matrix, source_hashes=sequence_length_source_hashes
        )
        config.max_seq_len = resolved_seq_len
        manifest["sequence_length_audit"] = seq_len_audit
        _json_write(manifest_path, manifest)
        parser_dir = ROOT / "data" / "parser" / f"stanza-{matrix['parser']['version']}"
        parser = IndonesianStanzaParser(parser_dir, allow_download=download_parser)
        parser_artifacts = parser.artifact_hashes()
        parser_fingerprint_payload = {"version": parser.version, "artifacts": parser_artifacts}
        legacy_parser_fingerprint = hashlib.sha256(
            json.dumps(
                parser_fingerprint_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        parser_fingerprint_payload.update(
            {
                "syntax_feature_schema": 2,
                "head_index_convention": "one-based-non-root-root-zero",
            }
        )
        parser_fingerprint = hashlib.sha256(
            json.dumps(
                parser_fingerprint_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        parser.configure_cache(
            ROOT / "data" / "cache" / f"stanza-{parser.version}-{parser_fingerprint[:16]}",
            legacy_cache_dirs=(
                ROOT / "data" / "cache" / f"stanza-{parser.version}-{legacy_parser_fingerprint[:16]}",
            ),
        )
        stage_dir, maps = _prepare_training_data(
            run_dir,
            {"train": train_records, "dev": dev_records},
        )

        observer: dict[str, Any] = {}
        config.model = EMCGCN if architecture == "emcgcn" else __import__(
            "aste_repro.ssgcn", fromlist=["build_ssgcn_class"]
        ).build_ssgcn_class()
        config.model_name = architecture
        config.pretrained_bert = str(resolved_model_dir)
        config.dataset_file = {
            "train": str(stage_dir / "train.txt"),
            "valid": str(stage_dir / "valid.txt"),
            "test": str(stage_dir / "test.txt"),
        }
        config.dataset_name = dataset.upper()
        config.task = "triplet"
        config.seed = int(seed)
        config.num_epoch = 1 if purpose == "smoke" else int(matrix["training"]["epochs"])
        config.epochs = config.num_epoch
        config.batch_size = int(matrix["training"]["batch_size"])
        config.max_seq_len = resolved_seq_len
        config.dynamic_truncate = bool(matrix["training"]["dynamic_truncate"])
        config.use_bert_spc = bool(matrix["training"]["use_bert_spc"])
        config.learning_rate = float(matrix["training"]["learning_rate"])
        config.l2reg = float(matrix["training"]["weight_decay"])
        config.emb_dropout = float(matrix["training"]["dropout"])
        config.initializer = "xavier_uniform_"
        config.output_dim = int(matrix["training"]["output_dim"])
        config.cross_validate_fold = -1
        config.cache_dataset = False
        config.overwrite_cache = True
        config.verbose = False
        config.use_amp = False
        config.relation_constraint = True
        config.symmetry_decoding = False
        config.num_layers = 1
        config.gcn_dim = int(matrix["ssgcn_paper_settings"]["gcn_dim"])
        config.ssgcn_attention_heads = 12
        config.warmup_step = -1
        config.evaluate_begin = 0
        config.patience = 100000
        config.log_step = max(1, (len(train_records) + config.batch_size - 1) // config.batch_size)
        config.save_mode = 1

        SafeTrainer, exact_dataset, _ = build_pyabsa_training_classes(
            parser=parser,
            validation_records=dev_records,
            validation_token_maps=maps["dev"],
            observer=observer,
        )
        # DatasetDict bypasses PyABSA's filename autodiscovery, which expects
        # its own task-code suffixes and otherwise discards this explicit path
        # map. The compatibility instructor reads only config.dataset_file.
        trainer_dataset = DatasetDict()
        trainer_dataset["dataset_name"] = config.dataset_name
        # The trainer's internal test set is the validation file above. No
        # held-out test file or test label is available before this returns.
        previous_cwd = Path.cwd()
        try:
            os.chdir(run_dir)
            with log_path.open("w", encoding="utf-8", newline="\n") as log_stream:
                with contextlib.redirect_stdout(log_stream), contextlib.redirect_stderr(log_stream):
                    trainer = SafeTrainer(
                        config=config,
                        dataset=trainer_dataset,
                        checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
                        auto_device=DeviceTypeOption.AUTO,
                        path_to_save=str(run_dir / "checkpoints"),
                        load_aug=False,
                    )
        finally:
            os.chdir(previous_cwd)

        checkpoint_dir = Path(trainer.inference_model).resolve()
        if not checkpoint_dir.is_dir():
            raise RuntimeError("validation-selected checkpoint directory was not created")
        if not observer.get("optimizer_groups"):
            raise RuntimeError("PyABSA optimizer group settings were not captured")

        # The test rows are loaded only now, after the model checkpoint and its
        # validation score are frozen. This is the sole real-test evaluation.
        test_records = load_hoasa(canonical_paths["test"])
        if purpose == "smoke":
            test_records = _copy_records_for_smoke(test_records, max(1, int(smoke_examples)))
        test_scores, artifacts = _external_test_evaluation(
            trainer=trainer,
            dataset_class=exact_dataset,
            records=test_records,
            run_dir=run_dir,
        )
        _json_write(run_dir / "metrics.json", test_scores)

        resolved = {
            "python": platform.python_version(),
            "packages": _module_versions(),
            "hardware": {
                "platform": platform.platform(),
                "torch_device": str(trainer.config.device),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
            },
            "encoder": model_ref,
            "encoder_snapshot": str(resolved_model_dir),
            "training_controls": matrix["training"],
            "resolved_training_controls": {
                "epochs": int(trainer.config.num_epoch),
                "batch_size": int(trainer.config.batch_size),
                "max_seq_len": int(trainer.config.max_seq_len),
                "max_seq_len_policy": seq_len_audit,
                "dynamic_truncate": bool(trainer.config.dynamic_truncate),
                "use_bert_spc": bool(trainer.config.use_bert_spc),
                "dropout": float(trainer.config.emb_dropout),
                "l2": float(trainer.config.l2reg),
                "initializer": trainer.config.initializer,
                "dataset_cache_requested_by_plan": bool(matrix["training"]["cache_dataset"]),
                "pyabsa_dataset_cache": bool(trainer.config.cache_dataset),
                "cache_deviation": "PyABSA feature caching is disabled; validated syntax parses use a separate hash-keyed cache.",
                "cross_validate_fold": int(trainer.config.cross_validate_fold),
                "precision": "float32; AMP disabled",
            },
            "randomness": deterministic,
            "lockfile_sha256": _sha256(ROOT / "uv.lock") if (ROOT / "uv.lock").is_file() else None,
            "parser": {
                "library": "stanza",
                "version": parser.version,
                "language": "id",
                "processors": matrix["parser"]["processors"],
                "tokenize_pretokenized": True,
                "head_index_convention": "one-based token index; root=0; PyABSA Instance subtracts one",
                "model_artifact_sha256": parser_artifacts,
                "model_fingerprint": parser_fingerprint,
                "syntax_cache": parser.cache_usage(),
            },
            "pyabsa": {
                "version": pyabsa_version,
                "graph_layer_memory_adaptation": "Factor the stock refining Linear projection across edge and node terms to avoid materializing a batch×sequence×sequence×hidden activation; parameter shapes and linear weights are unchanged.",
                "output_dim": int(trainer.config.output_dim),
                "output_labels": list(trainer.config.label_to_index),
                "optimizer_groups": _resolved_optimizer_groups(observer),
                "actual_selection_metric": "canonical exact triplet micro-F1 including implicit gold",
                "internal_test_alias": "validation",
            },
            "evaluator": {
                "name": "aste_repro.metrics.evaluate_triplets",
                "version": 1,
                "aggregation": "set-based exact-match micro counts by sentence, aspect span, opinion span, and polarity",
                "primary_implicit_aspect_policy": "retained in canonical gold and test denominator",
                "secondary_aspect_term_policy": "implicit aspects are excluded",
            },
            "ssgcn": {
                "paper_settings": matrix["ssgcn_paper_settings"],
                "shared_matrix_epochs": int(matrix["training"]["epochs"]),
                "attention_heads": int(config.ssgcn_attention_heads),
                "distance_cutoffs": list(range(1, int(config.ssgcn_attention_heads) + 1)),
                "pyabsa_objective_weights": {
                    "primary_triplet_tag_loss": 1.0,
                    "interactive_biaffine": 0.1,
                    "relative_position": 0.01,
                    "dependency_relation": 0.01,
                    "pos_pair": 0.01,
                    "tree_distance": 0.01,
                },
                "adaptation_notes": [
                    "PyABSA's ten ASTE tag channels are used for each of the five graph relations.",
                    "Aspect attention is computed for every token pair because the PyABSA grid decoder predicts spans jointly.",
                    "The shared benchmark dropout and epoch settings, Indonesian parser, and PyABSA tag/decoder contract differ from the paper setup.",
                ],
                "gcn_relation_channels": [
                    "syntax-semantics interaction",
                    "relative distance",
                    "dependency relation",
                    "POS pair",
                    "dependency-tree distance",
                ],
            }
            if architecture == "ssgcn"
            else None,
        }
        manifest.update(
            {
                "status": "smoke" if purpose == "smoke" else "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "model": model_ref,
                "resolved_config": resolved,
                "selected_checkpoint": str(checkpoint_dir.relative_to(run_dir)),
                "validation_selection_f1": max(
                    float(item["f1"]) for item in observer["validation_history"]
                ),
                "validation_history": observer["validation_history"],
                "heldout_test_evaluations": 1,
                "artifacts": {
                    "checkpoint_dir": str(checkpoint_dir.relative_to(run_dir)),
                    "metrics": str((run_dir / "metrics.json").relative_to(run_dir)),
                    "predictions": str(artifacts["predictions"].relative_to(run_dir)),
                    "training_log": str(log_path.relative_to(run_dir)),
                },
            }
        )
        artifact_hashes = {}
        for path in sorted(item for item in run_dir.rglob("*") if item.is_file() and item != manifest_path):
            artifact_hashes[path.relative_to(run_dir).as_posix()] = _sha256(path)
        manifest["artifact_sha256"] = artifact_hashes
        _json_write(manifest_path, manifest)
        return manifest
    except KeyboardInterrupt:
        manifest.update(
            {
                "status": "interrupted",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "failure_type": "KeyboardInterrupt",
                "failure": "run interrupted before completion; no benchmark score was recorded",
            }
        )
        _json_write(manifest_path, manifest)
        raise
    except Exception as error:
        manifest.update(
            {
                "status": "blocked" if isinstance(error, RunBlocked) else "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "failure_type": type(error).__name__,
                "failure": str(error)[:1000],
            }
        )
        _json_write(manifest_path, manifest)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["hoasa", "casa"], required=True)
    parser.add_argument("--architecture", choices=["emcgcn", "ssgcn"], required=True)
    parser.add_argument("--encoder", choices=["mbert", "indobert", "xlmr", "deberta_absa"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--purpose", choices=["smoke", "benchmark"], default="benchmark")
    parser.add_argument("--smoke-examples", type=int, default=24)
    parser.add_argument("--download-parser", action="store_true", help="download missing Indonesian Stanza models")
    args = parser.parse_args()
    try:
        result = run_experiment(
            dataset=args.dataset,
            architecture=args.architecture,
            encoder=args.encoder,
            seed=args.seed,
            purpose=args.purpose,
            smoke_examples=args.smoke_examples,
            download_parser=args.download_parser,
        )
    except RunBlocked as error:
        print(f"run blocked: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"run failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "validation_selection_f1": result.get("validation_selection_f1"),
                "heldout_test_evaluations": result.get("heldout_test_evaluations"),
            },
            sort_keys=True,
        )
    )
    return 0


def download_parser_main() -> int:
    parser = argparse.ArgumentParser(description="Download the pinned Indonesian Stanza parser models.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "parser" / "stanza-1.14.0",
    )
    args = parser.parse_args()
    try:
        hashes = download_indonesian_parser(args.output)
    except Exception as error:
        print(f"parser setup failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"model_dir": str(args.output), "files": len(hashes), "sha256": hashes}, sort_keys=True))
    return 0
