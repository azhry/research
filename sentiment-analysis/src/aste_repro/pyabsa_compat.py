"""Guarded data, trainer, and evaluation adapters for PyABSA 2.4.3."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .hoasa import HoasaRecord, map_model_span_to_source
from .metrics import evaluate_triplets, normalize_triplet

PARSER_CACHE_SCHEMA = 2


class IndonesianStanzaParser:
    """Stanza parser adapter with pre-tokenization and strict token alignment."""

    def __init__(self, model_dir: Path, allow_download: bool = False) -> None:
        try:
            import stanza
        except ImportError as error:
            raise RuntimeError("install the locked project dependencies to use Stanza") from error

        self.model_dir = model_dir.resolve()
        self.cache_dir: Path | None = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_migrations = 0
        self.cache_files: dict[str, str] = {}
        self.legacy_cache_dirs: list[Path] = []
        id_dir = self.model_dir / "id"
        if not id_dir.exists():
            if not allow_download:
                raise RuntimeError(
                    "Indonesian Stanza models are missing; run `aste-download-parser` first"
                )
            stanza.download(
                "id",
                model_dir=str(self.model_dir),
                processors={
                    "tokenize": "default",
                    "pos": "default",
                    "lemma": "default",
                    "depparse": "default",
                },
                verbose=False,
            )
        try:
            self.pipeline = stanza.Pipeline(
                lang="id",
                processors="tokenize,pos,lemma,depparse",
                tokenize_pretokenized=True,
                dir=str(self.model_dir),
                download_method=None,
                use_gpu=False,
                verbose=False,
            )
        except Exception as error:
            raise RuntimeError(
                "could not initialize pinned Indonesian Stanza processors"
            ) from None
        self.version = stanza.__version__

    def __call__(self, text: str) -> tuple[list[str], list[int], list[str]]:
        source_tokens = text.split()
        cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / cache_key[:2] / f"{cache_key}.json" if self.cache_dir else None
        if cache_path and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                pos = cached["pos"]
                heads = cached["heads"]
                dependencies = cached["dependencies"]
                if (
                    cached.get("schema_version") != PARSER_CACHE_SCHEMA
                    or cached.get("cache_key") != cache_key
                    or cached.get("token_count") != len(source_tokens)
                    or not isinstance(pos, list)
                    or not isinstance(heads, list)
                    or not isinstance(dependencies, list)
                    or len(pos) != len(source_tokens)
                    or len(heads) != len(source_tokens)
                    or len(dependencies) != len(source_tokens)
                    or not all(isinstance(value, str) for value in pos + dependencies)
                    or not all(type(value) is int for value in heads)
                ):
                    raise ValueError
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                raise ValueError("cached Indonesian parse is malformed; refusing to overwrite it") from None
            self.cache_hits += 1
            self.cache_files[cache_key] = hashlib.sha256(cache_path.read_bytes()).hexdigest()
            return pos, heads, dependencies
        legacy_path = next(
            (
                legacy_dir / cache_key[:2] / f"{cache_key}.json"
                for legacy_dir in self.legacy_cache_dirs
                if (legacy_dir / cache_key[:2] / f"{cache_key}.json").is_file()
            ),
            None,
        )
        if legacy_path:
            try:
                legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
                old_pos = legacy["pos"]
                old_heads = legacy["heads"]
                old_dependencies = legacy["dependencies"]
                if (
                    legacy.get("schema_version") != 1
                    or legacy.get("cache_key") != cache_key
                    or legacy.get("token_count") != len(source_tokens)
                    or not isinstance(old_pos, list)
                    or not isinstance(old_heads, list)
                    or not isinstance(old_dependencies, list)
                    or len(old_pos) != len(source_tokens)
                    or len(old_heads) != len(source_tokens)
                    or len(old_dependencies) != len(source_tokens)
                    or not all(isinstance(value, str) for value in old_pos + old_dependencies)
                    or not all(type(value) is int for value in old_heads)
                ):
                    raise ValueError
                # Schema 1 stored Stanza's 1-based heads after subtracting one.
                # The original root head (0) is distinguishable by its `root`
                # dependency; restore every non-root head to the 1-based form
                # consumed by PyABSA Instance (which itself subtracts one).
                heads = [
                    0 if dependency == "root" else head + 1
                    for head, dependency in zip(old_heads, old_dependencies)
                ]
                if any(head < 0 or head > len(source_tokens) for head in heads):
                    raise ValueError
                pos = old_pos
                dependencies = old_dependencies
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                raise ValueError("legacy cached Indonesian parse is malformed; refusing to overwrite it") from None
            self._write_cache(cache_path, cache_key, pos, heads, dependencies)
            self.cache_hits += 1
            self.cache_migrations += 1
            self.cache_files[cache_key] = hashlib.sha256(cache_path.read_bytes()).hexdigest()
            return pos, heads, dependencies
        try:
            document = self.pipeline(text)
            if len(document.sentences) != 1:
                raise ValueError
            words = document.sentences[0].words
            if len(words) != len(source_tokens):
                raise ValueError
            parsed_tokens = [word.text for word in words]
            if parsed_tokens != source_tokens:
                raise ValueError
            pos = [word.upos or "X" for word in words]
            # Stanza and PyABSA Instance use 1-based heads, with root head 0.
            # Instance subtracts one when converting each non-root head index.
            heads = [word.head for word in words]
            dependencies = [word.deprel or "dep" for word in words]
        except Exception:
            raise ValueError("Indonesian parse failed or changed whitespace-token alignment") from None
        if not (len(pos) == len(heads) == len(dependencies) == len(source_tokens)):
            raise ValueError("Indonesian parse returned inconsistent token annotations")
        if cache_path:
            self._write_cache(cache_path, cache_key, pos, heads, dependencies)
            self.cache_misses += 1
            self.cache_files[cache_key] = hashlib.sha256(cache_path.read_bytes()).hexdigest()
        return pos, heads, dependencies

    @staticmethod
    def _write_cache(
        cache_path: Path,
        cache_key: str,
        pos: list[str],
        heads: list[int],
        dependencies: list[str],
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached = {
            "schema_version": PARSER_CACHE_SCHEMA,
            "cache_key": cache_key,
            "token_count": len(pos),
            "pos": pos,
            "heads": heads,
            "dependencies": dependencies,
        }
        temporary = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(cached, separators=(",", ":"), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, cache_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def configure_cache(
        self, cache_dir: Path, *, legacy_cache_dirs: tuple[Path, ...] = ()
    ) -> None:
        self.cache_dir = cache_dir.resolve()
        self.legacy_cache_dirs = [path.resolve() for path in legacy_cache_dirs]
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_usage(self) -> dict[str, Any]:
        digest = hashlib.sha256()
        for key, value in sorted(self.cache_files.items()):
            digest.update(key.encode("ascii"))
            digest.update(value.encode("ascii"))
        return {
            "entries_used": len(self.cache_files),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "migrations": self.cache_migrations,
            "schema_version": PARSER_CACHE_SCHEMA,
            "used_entries_sha256": digest.hexdigest(),
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
        }

    def artifact_hashes(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for path in sorted(self.model_dir.rglob("*")):
            if path.is_file():
                relative = path.relative_to(self.model_dir).as_posix()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                result[relative] = digest
        return result


def download_indonesian_parser(model_dir: Path) -> dict[str, str]:
    parser = IndonesianStanzaParser(model_dir, allow_download=True)
    return parser.artifact_hashes()


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return list(value)


def predict_triplets(
    model: Any,
    data_loader: Any,
    config: Any,
    records: list[HoasaRecord],
    token_maps: dict[int, list[int]],
) -> tuple[list[list[tuple[tuple[int, ...], tuple[int, ...], str]]], dict[str, int]]:
    """Decode PyABSA's grid to source token indices without text re-alignment."""
    import torch
    import torch.nn.functional as F
    from pyabsa.tasks.AspectSentimentTripletExtraction.dataset_utils.aste_utils import Metric

    all_ids: list[int] = []
    all_predictions = []
    all_labels = []
    all_lengths: list[int] = []
    all_sentence_lengths: list[int] = []
    all_token_ranges: list[list[list[int]]] = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            (
                sentence_ids,
                _sentences,
                token_ids,
                lengths,
                masks,
                sentence_lengths,
                token_ranges,
                _aspect_tags,
                tags,
                word_pair_position,
                word_pair_deprel,
                word_pair_pos,
                word_pair_synpost,
                _tags_symmetry,
            ) = batch
            inputs = {
                "token_ids": token_ids,
                "masks": masks,
                "word_pair_position": word_pair_position,
                "word_pair_deprel": word_pair_deprel,
                "word_pair_pos": word_pair_pos,
                "word_pair_synpost": word_pair_synpost,
            }
            logits = model(inputs)[-1]
            all_predictions.append(torch.argmax(F.softmax(logits, dim=-1), dim=3))
            all_labels.append(tags)
            all_ids.extend(int(item) for item in _as_list(sentence_ids))
            all_lengths.extend(int(item) for item in _as_list(lengths))
            all_sentence_lengths.extend(int(item) for item in _as_list(sentence_lengths))
            all_token_ranges.extend(
                item.detach().cpu().tolist() if hasattr(item, "detach") else item
                for item in token_ranges
            )

    if not all_predictions:
        raise ValueError("PyABSA produced no evaluation batches")
    predictions = torch.cat(all_predictions, dim=0).cpu().tolist()
    labels = torch.cat(all_labels, dim=0).cpu().tolist()
    golden_and_predicted = Metric(
        config,
        predictions,
        labels,
        all_lengths,
        all_sentence_lengths,
        all_token_ranges,
    )
    _, decoded = golden_and_predicted.parse_triplet(golden=False)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("evaluation data loader repeated a source row")
    if set(all_ids) != {record.row_id for record in records}:
        raise ValueError("evaluation data loader changed source row coverage")

    sentiment_by_id = {7: "NEG", 8: "NEU", 9: "POS"}
    predictions_by_id: dict[int, list[tuple[tuple[int, ...], tuple[int, ...], str]]] = {}
    unmapped = 0
    for row_id, row_predictions in zip(all_ids, decoded):
        offset_map = token_maps[row_id]
        mapped_row = []
        for aspect_start, aspect_end, opinion_start, opinion_end, sentiment_id in row_predictions:
            aspect = map_model_span_to_source(aspect_start, aspect_end, offset_map)
            opinion = map_model_span_to_source(opinion_start, opinion_end, offset_map)
            sentiment = sentiment_by_id.get(int(sentiment_id))
            if aspect is None or opinion is None or sentiment is None:
                unmapped += 1
                continue
            mapped_row.append((aspect, opinion, sentiment))
        predictions_by_id[row_id] = mapped_row

    ordered_records = sorted(records, key=lambda record: record.row_id)
    if [record.row_id for record in ordered_records] != list(range(len(ordered_records))):
        raise ValueError("canonical rows must have contiguous, zero-based source identifiers")
    ordered_predictions = [predictions_by_id[record.row_id] for record in ordered_records]
    return ordered_predictions, {
        "sentences": len(all_ids),
        "predicted_triplets_unmapped": unmapped,
    }


def _write_exact_dataset_class(parser: IndonesianStanzaParser):
    """Build a stock-compatible adapter with corrected triplet and parser behavior."""
    from pyabsa.tasks.AspectSentimentTripletExtraction.dataset_utils.data_utils_for_training import (
        ASTEDataset as StockASTEDataset,
        generate_tags,
        load_tokens,
    )

    class ExactIndonesianASTEDataset(StockASTEDataset):
        def __init__(self, config, tokenizer, dataset_type="train"):
            # PyABSA keeps these on the class by default, leaking vocab entries
            # across split instances. Instance-local lists keep vocab train-only.
            self.all_tokens = []
            self.all_deprel = []
            self.all_postag = []
            self.all_postag_ca = []
            self.all_max_len = []
            self.nlp = parser
            super(StockASTEDataset, self).__init__(
                config=config,
                tokenizer=tokenizer,
                dataset_type=dataset_type,
            )
            self.config.label_to_index = self.label_to_index
            self.config.index_to_label = self.index_to_label
            self.config.output_dim = len(self.label_to_index)

        def get_syntax_annotation(self, sentence, annotation):
            tokens = sentence.split()
            postags, heads, deprels = self.get_dependencies(tokens)
            if not (len(tokens) == len(postags) == len(heads) == len(deprels)):
                raise ValueError("dependency parser changed whitespace-token alignment")
            triples = []
            for triplet_index, raw_triplet in enumerate(annotation):
                aspect, opinion, sentiment = normalize_triplet(raw_triplet)
                if aspect == (-1,):
                    continue
                if aspect[-1] >= len(tokens) or opinion[-1] >= len(tokens):
                    raise ValueError("triplet token index exceeds sentence length")
                target_tags = generate_tags(tokens, aspect[0], aspect[-1], "BIO")
                opinion_tags = generate_tags(tokens, opinion[0], opinion[-1], "BIO")
                labels = {"POS": "Positive", "NEG": "Negative", "NEU": "Neutral"}
                triples.append(
                    {
                        "uid": str(triplet_index),
                        "target_tags": target_tags,
                        "opinion_tags": opinion_tags,
                        "sentiment": labels[sentiment],
                    }
                )
            return {
                "id": "",
                "sentence": sentence,
                "postag": postags,
                "head": heads,
                "deprel": deprels,
                "triples": triples,
            }

        def get_dependencies(self, tokens):
            postags, heads, deprels = self.nlp(" ".join(tokens))
            return postags, heads, deprels

        def load_data_from_file(self, file_path, **kwargs):
            dataset_path = Path(self.config.dataset_file[self.dataset_type])
            all_data = []
            try:
                lines = dataset_path.read_text(encoding="utf-8-sig").splitlines()
            except OSError as error:
                raise ValueError(f"cannot read PyABSA {self.dataset_type} data file") from error
            for row_id, line in enumerate(lines):
                try:
                    if line.count("####") != 1:
                        raise ValueError
                    sentence, raw_triplets = line.split("####", maxsplit=1)
                    sentence = sentence.strip()
                    annotations = ast.literal_eval(raw_triplets.strip())
                    if not sentence or not isinstance(annotations, list):
                        raise ValueError
                    prepared = self.get_syntax_annotation(sentence, annotations)
                    prepared["id"] = row_id
                    prepared["sentence"] = sentence.replace("placeholder", "-")
                    tokens, dependencies, pos_pairs, pos, max_len = load_tokens(prepared)
                    self.all_tokens.extend(tokens)
                    self.all_deprel.extend(dependencies)
                    self.all_postag.extend(pos_pairs)
                    self.all_postag_ca.extend(pos)
                    self.all_max_len.append(max_len)
                    all_data.append(prepared)
                except Exception:
                    raise ValueError(
                        f"{dataset_path.name}:{row_id + 1}: data or parser alignment failed"
                    ) from None
            self.data = all_data

        def convert_examples_to_features(self):
            self.get_vocabs()
            from pyabsa.tasks.AspectSentimentTripletExtraction.dataset_utils.aste_utils import (
                Instance,
            )

            features = []
            for row_id, data in enumerate(self.data):
                try:
                    features.append(
                        Instance(
                            self.tokenizer,
                            data,
                            self.config.post_vocab,
                            self.config.deprel_vocab,
                            self.config.postag_vocab,
                            self.config.syn_post_vocab,
                            self.config,
                        )
                    )
                except Exception as error:
                    raise ValueError(
                        f"{self.dataset_type} row {row_id}: tokenizer/feature conversion failed "
                        f"({type(error).__name__})"
                    ) from None
            self.data = features

    return ExactIndonesianASTEDataset


def build_pyabsa_training_classes(
    *,
    parser: IndonesianStanzaParser,
    validation_records: list[HoasaRecord],
    validation_token_maps: dict[int, list[int]],
    observer: dict[str, Any],
):
    """Create an instructor that never routes the held-out test to PyABSA."""
    from pyabsa.tasks.AspectSentimentTripletExtraction.instructor.instructor import (
        ASTETrainingInstructor,
    )
    from pyabsa.tasks.AspectSentimentTripletExtraction.trainer.trainer import ASTETrainer
    from pyabsa.framework.tokenizer_class.tokenizer_class import PretrainedTokenizer

    exact_dataset = _write_exact_dataset_class(parser)

    class ValidationOnlyInstructor(ASTETrainingInstructor):
        def __init__(self, config):
            super().__init__(config)
            observer["resolved_output_dim"] = int(self.config.output_dim)
            optimizer = getattr(self, "optimizer", None)
            if optimizer is None:
                raise RuntimeError("PyABSA did not initialize its optimizer before training")
            observer["optimizer_groups"] = [
                {
                    "learning_rate": float(group["lr"]),
                    "weight_decay": float(group.get("weight_decay", self.config.l2reg)),
                    "parameter_tensors": len(group["params"]),
                    "parameters": sum(int(parameter.numel()) for parameter in group["params"]),
                }
                for group in optimizer.param_groups
            ]

        def _load_dataset_and_prepare_dataloader(self):
            self.tokenizer = PretrainedTokenizer(self.config)
            self.train_set = exact_dataset(self.config, self.tokenizer, dataset_type="train")
            self.valid_set = exact_dataset(self.config, self.tokenizer, dataset_type="valid")
            # PyABSA's internal test slot is a reference to validation. The
            # real held-out test path is opened only after trainer completion.
            self.test_set = self.valid_set
            self.train_set.convert_examples_to_features()
            self.valid_set.convert_examples_to_features()
            self.model = self.config.model(config=self.config).to(self.config.device)
            install_memory_efficient_graph_layers(self.model)
            self.config.tokenizer = self.tokenizer

        def _train(self, criterion):
            self._prepare_dataloader()
            if len(self.valid_dataloaders) != 1 or self.test_dataloader is None:
                raise RuntimeError("validation-only training requires one dev loader")
            # The stock _train dispatch interprets two validation loaders as
            # k-fold CV. Bypass that branch and let _train_and_evaluate select
            # on validation loader zero, with its test alias still on dev.
            self.valid_dataloaders.append(self.valid_dataloaders[0])
            return self._train_and_evaluate(criterion)

        def _evaluate_f1(self, data_loader, FLAG=False):
            predicted, decode_diagnostics = predict_triplets(
                self.model,
                data_loader,
                self.config,
                validation_records,
                validation_token_maps,
            )
            gold = [record.triplets for record in sorted(validation_records, key=lambda item: item.row_id)]
            scores = evaluate_triplets(gold, predicted)
            metrics = scores["triplet"]
            event = {
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "sentence_count": scores["sentence_count"],
                "gold_triplets": metrics["gold"],
                **decode_diagnostics,
            }
            observer.setdefault("validation_history", []).append(event)
            return event["precision"], event["recall"], event["f1"]

    class SafeASTETrainer(ASTETrainer):
        def _run(self):
            self.training_instructor = ValidationOnlyInstructor
            super()._run()

    return SafeASTETrainer, exact_dataset, ValidationOnlyInstructor


def read_token_maps(path: Path) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line)
            row_id = int(value["row_id"])
            if row_id in result:
                raise ValueError(f"duplicate token-map row {row_id} at line {line_number}")
            result[row_id] = [int(index) for index in value["model_token_to_source"]]
    return result


def install_memory_efficient_graph_layers(model: Any) -> None:
    """Replace PyABSA's large concatenation with its algebraically equivalent linear sum.

    PyABSA materializes two [batch, seq, seq, hidden] node-pair tensors before
    applying one Linear layer. At the full HoASA sequence capacity that exceeds
    available CPU memory. Splitting the Linear weight by concatenation segment
    and projecting each node/edge component before broadcasting preserves the
    same parameters and mathematical operation without the hidden-width pair
    tensor.
    """
    import torch
    from torch.nn import functional as F
    from pyabsa.tasks.AspectSentimentTripletExtraction.models.model import GraphConvLayer

    class MemoryEfficientGraphConvLayer(GraphConvLayer):
        def forward(self, weight_prob_softmax, weight_adj, gcn_inputs, self_loop):
            batch, seq, dim = gcn_inputs.shape
            edge_dim = self.edge_dim
            weight_prob_softmax = weight_prob_softmax.permute(0, 3, 1, 2)
            expanded_nodes = gcn_inputs.unsqueeze(1).expand(batch, edge_dim, seq, dim)
            weight_prob_softmax += self_loop
            aggregated = torch.matmul(weight_prob_softmax, expanded_nodes)
            if self.pooling == "avg":
                aggregated = aggregated.mean(dim=1)
            elif self.pooling == "max":
                aggregated, _ = aggregated.max(dim=1)
            elif self.pooling == "sum":
                aggregated = aggregated.sum(dim=1)
            else:
                raise ValueError(f"unsupported PyABSA graph pooling: {self.pooling}")

            node_outputs = F.relu(self.layernorm(self.W(aggregated)))
            weight_prob_softmax = weight_prob_softmax.permute(0, 2, 3, 1).contiguous()

            strategy = self.highway
            linear = strategy.W
            hidden_dim = strategy.hidden_dim
            if linear.in_features != 3 * edge_dim + 2 * hidden_dim:
                raise RuntimeError("PyABSA refining-strategy input layout changed")
            weight = linear.weight
            edge_diag = torch.diagonal(weight_adj, offset=0, dim1=1, dim2=2)
            edge_diag = edge_diag.permute(0, 2, 1).contiguous()

            # Original concatenation order is [edge, diag[j], diag[i], node[j], node[i]].
            split = 0
            edge_weight = weight[:, split : split + edge_dim]
            split += edge_dim
            edge_i_weight = weight[:, split : split + edge_dim]
            split += edge_dim
            edge_j_weight = weight[:, split : split + edge_dim]
            split += edge_dim
            node_j_weight = weight[:, split : split + hidden_dim]
            split += hidden_dim
            node_i_weight = weight[:, split : split + hidden_dim]

            refined_edges = F.linear(weight_adj, edge_weight, linear.bias)
            refined_edges = refined_edges + F.linear(edge_diag.unsqueeze(1), edge_i_weight)
            refined_edges = refined_edges + F.linear(edge_diag.unsqueeze(2), edge_j_weight)
            refined_edges = refined_edges + F.linear(node_outputs.unsqueeze(1), node_j_weight)
            refined_edges = refined_edges + F.linear(node_outputs.unsqueeze(2), node_i_weight)
            return node_outputs, refined_edges

    replacements = []
    model_device = next(model.parameters()).device
    for original in model.gcn_layers:
        replacement = MemoryEfficientGraphConvLayer(
            original.device,
            original.gcn_dim,
            original.edge_dim,
            original.dep_embed_dim,
            original.pooling,
        )
        replacement.load_state_dict(original.state_dict(), strict=True)
        replacements.append(replacement.to(model_device))
    model.gcn_layers = torch.nn.ModuleList(replacements)
