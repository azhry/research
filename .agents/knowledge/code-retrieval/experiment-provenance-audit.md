# Submitted experiment provenance and evaluation splits

Audit date: 2026-10-13; reviewer verdict mapping added 2026-10-14
Study: CoIR-Retrieval/CosQA with E5, HyDE, and cross-encoder reranking
Scope: retrospective inspection of the existing pinned data, selection artifacts, saved runs, evaluator source, and paper revisions. No model inference, benchmark rerun, PDF rebuild, or test suite was run.

## Findings

- The current saved selection and evaluation populations are disjoint: each has 500 queries and 500 positive qrel rows. Query IDs, normalized query text, positive judgments, relevant documents, and within-split duplicate groups all have zero overlap.
- The frozen selection manifest uses valid-only nDCG@10, candidate depth 1,000, RRF k=60, and a 21-point grid at 0.05 increments. All four ranking input hashes and the reranker validation-summary hash match their local files.
- Selected E5/component weights are 0.65/0.35 for E5+HyDE, 0.80/0.20 for E5+reranker, and 0.75/0.25 for the combined system. The saved final run records those weights with test qrels.
- The saved local FAISS baseline is 0.45876 nDCG@10. A separate cached-vector CoIR path check is 0.3259. Retrieval and tie-order handling differ, so the saved evidence does not isolate a single cause.
- The exact submitted PDF remains unverified. Revision 28 is the strongest local anonymous candidate, but there is no receipt or preserved upload to establish it was submitted.
- The final run’s Git commit exists. Its recorded notebook hashes do not match available notebook files, so exact executed notebook bytes remain unverified.

## Claim-to-artifact map

| Concern | Evidence | Verdict |
| --- | --- | --- |
| Selection/test query overlap | Pinned valid/test qrels; split hashes below. | Zero shared IDs, normalized texts, judgments, or relevant documents in this local artifact chain. |
| Duplicate query text | Unicode NFKC, casefold, collapsed/trimmed whitespace grouping. | Zero duplicate groups within either split. |
| Weights tuned on test | Selection manifest, validation metadata, final run manifest. | Selection split is valid; saved final run split is test. |
| Selection on all 500 valid queries | Selection and reranker validation manifests. | Confirmed. Valid scores are tuning results, not held-out claims. |
| Exact submitted paper | Revision snapshots, local PDFs, README; no EDAS receipt or submission copy. | Unverified; revision 28 is only the best local candidate. |
| CoIR versus local FAISS | Cached CoIR result, local run, installed CoIR package source. | Protocols differ; score gap is measured after encoding but its cause is unresolved. |
| Executed notebook identity | Run hashes compared with repository/run notebook files. | Commit identity is known; byte-level executed notebook identity is unverified. |

## Reviewer concern findings

Scope: this audit covers R3's split/weight-selection and optimistic-bias concerns and R2's implementation-detail concern. These verdicts distinguish later local evidence from the conference upload, whose exact file remains unidentified.

### R3 — split reuse, weight selection, and strength of evidence

| Concern | Verdict | Evidence and interpretation |
| --- | --- | --- |
| The 500 tuning queries were reused as the 500 test queries. | Contradicted by the saved later artifact chain; unresolved for the conference upload. | The pinned CoIR-Retrieval/cosqa revision has 500 `valid` and 500 `test` queries. Their IDs, normalized texts, positive judgments, and relevant documents have zero overlap. The selection manifest uses `valid`; the saved final run uses `test`; the ranking and validation-summary hashes match. The revision-28 anonymous manuscript states this separation. The revision-00 PDF is explicitly a review input, not proof of the conference upload. |
| Weights were selected using all 500 validation queries. | Confirmed. | The frozen manifest selects on all 500 `valid` queries, maximizing nDCG@10 over 21 RRF weights from 0.00 to 1.00 by 0.05, with k=60. Therefore validation scores are tuned selection results and must not be presented as held-out performance. This is distinct from reusing test queries. |
| Test judgments influenced the saved weight selection. | Not supported by the saved later artifact chain; unresolved for the conference upload. | The manifest records `valid`, its input hashes match, and the final saved evaluation uses the disjoint `test` split with those frozen weights. No local artifact inspected shows test qrels entering selection. |
| The gains are small and do not establish every component's benefit. | Partly confirmed, with a clearer split-specific result. | On the saved 500-query test run, nDCG@10 changes versus E5 are: HyDE +0.01416 (95% paired-bootstrap CI [0.00313, 0.02523]); reranking +0.00209 ([-0.00949, 0.01348]); combined +0.01945 ([0.00894, 0.03017]). The reranking-only interval includes zero. Combined versus HyDE is +0.00529 ([-0.00024, 0.01137]), also including zero, so the best combined point score does not show an incremental gain over HyDE. Intervals use 20,000 paired query bootstrap replicates, seed 518, and are unadjusted across comparisons. |

The local evidence therefore supports positive nDCG@10 differences for HyDE and the combined system versus E5 on this saved test split, but not for reranking alone or for combined versus HyDE. The evidence remains one CosQA snapshot and one 500-query test sample; it does not establish broader code-search gains. It also cannot establish whether the conference submission used this exact data and selection chain.

### R2 — implementation detail and presentation

| Concern | Verdict | Evidence and interpretation |
| --- | --- | --- |
| Implementation description needs more detail. | Addressed in the later local manuscript; not verified for the conference upload. | The preserved Methods section specifies text-only CoIR CosQA inputs, E5 model/input settings and depth, the generation prompt and token limit, reranker and candidate pool, weighted-RRF equation, validation grid/metric/tie rule, deterministic expansion candidate union/truncation, and reranker candidate preservation. Revision 03's notes record the added validation-only selection and CoIR-path distinction; revision 28 retains this protocol. |
| References should appear in sequence. | Sequential in the revision-28 local PDF. | Its bibliography contains entries [1]–[12] in order, consistent with first citation order in `main.tex`. This checks the preserved local revision only. |
| Grammar needs checking. | Not conclusively adjudicated. | The reviewer supplied no specific sentence-level defects, and the preserved review notes do not record an independent full copyedit against that comment. |

### Submission identity boundary

`paper/README.md` identifies revision 28's anonymous PDF as the intended EDAS upload source, but the named `submission-anonymous.pdf` is absent. The root review PDF is byte-identical to revision 28's anonymous PDF; that proves the local copy relationship, not that EDAS received it. Revision 00 is labeled as material supplied to the review thread, not as a submission receipt. No EDAS receipt or exact uploaded file was found. Accordingly, R3's original-submission leakage allegation and R2's original-submission presentation assessment remain unverified even where later local revisions address them.

## Dataset and split audit

Both splits were loaded through the existing BaselineConfig and load_cosqa entry point. Dataset: CoIR-Retrieval/cosqa at revision 0846fa3b963a21bead36e9fab61451fc83b777a6. The query config/split is queries/queries, corpus is corpus/corpus, and qrels use config default with split valid or test. Inputs are text-only; titles and other metadata are excluded.

Normalization: Unicode NFKC, casefold, collapse and trim whitespace. Raw query text and IDs are not copied into this report.

| Measure | valid selection | test evaluation |
| --- | ---: | ---: |
| Queries / qrels queries | 500 / 500 | 500 / 500 |
| Qrel rows / positive judgments | 500 / 500 | 500 / 500 |
| Within-split duplicate normalized-text groups | 0 | 0 |
| Query-ID SHA-256 | 5711953f0c22fd6bbebf7095d61bf2cb52517ccb8f7c85d9084487b0e244c4fe | a29f4e7de6f5c87de8d1db8b419f06957810975f8d617f23e4961ffd1aa9caa3 |

Cross-split counts: shared IDs 0; normalized text groups 0; identical query/document/score positive judgments 0; normalized-query/document/score positive judgments 0; shared relevant document IDs 0. The reranker validation summary’s query-ID hash equals the valid loader hash.

This establishes separation for the current local artifact chain only. It does not establish which data artifacts appeared in the conference upload.

## Selection and final run

Selection manifest: code-retrieval/artifacts/e5_hyde_rerank/validation/weight_selection.json
SHA-256: aef1dda81f7cd8c207e823dc50a4314ce73bffcffed9be8a26ac215f66a98ef2

Selection controls: valid qrels, nDCG@10, 500 queries/500 judgments, 20,604 corpus documents, depth 1,000, RRF k=60, 21 weights from 0.00 to 1.00 by 0.05. Ties prefer the larger E5 weight. Evaluator: coir.beir.retrieval.evaluation.EvaluateRetrieval, coir-eval 0.7.0; saved ranking order is descending score then insertion order, evaluated with unique ordinal scores.

| System | Selected E5/component weights | Valid nDCG@10 | Final test nDCG@10 |
| --- | ---: | ---: | ---: |
| E5 + HyDE | 0.65/0.35 | 0.45683 | 0.47292 |
| E5 + reranker | 0.80/0.20 | 0.45177 | 0.46085 |
| E5 + HyDE + reranker | 0.75/0.25 | 0.45932 | 0.47821 |

Saved test baseline E5 is 0.45876 and E5+reranker is 0.46085. Final run ID is 20260930T103232-1790764352158657000-80d721673d195f47-metrics-v1; cache identity 80d721673d195f47; commit 92c128695c2d5b2da7d1bd7e3d7d7feb53cfcc9b. It records 500 test queries, 20,604 documents, 500 qrels queries, depth 1,000, and the same frozen weights. This is historical saved-run evidence; it was not rerun for the audit.

Recorded ranking input hashes all matched:

| Input | Path | SHA-256 |
| --- | --- | --- |
| E5 for HyDE | artifacts/e5_hyde/cache/c12cb9327cd85c39/original_rankings.json | 0bf6f0dcfb9c1ad90f6a567ac7a66b49fb608e344dd5b910a625f873c35dcd57 |
| HyDE query-plus | artifacts/e5_hyde/cache/c12cb9327cd85c39/rankings.json | 3f0677eca3c239602b05c16877f93ba4fd983a2c80f51a7413df1bdaef00dac7 |
| E5 for reranker | artifacts/e5_hyde_rerank/validation/rerank_depth1000/e5_rankings.json | 935041bc910df376cf7d4e5ac30a23e4f577165cfd6babe17f52c699a0493754 |
| Raw reranker | artifacts/e5_hyde_rerank/validation/rerank_depth1000/e5_reranked.json | f61138edad1ba128ac6bdca6f8137b06a0e2a9792dad40be6fda4de5c6a3af31 |

The reranker validation summary SHA-256 is 225b1e665abf33571f363a390dc407033bf3beb38d426de29d65666d240f73cd; its saved hash matches. Its query-ID hash is 5711953f0c22fd6bbebf7095d61bf2cb52517ccb8f7c85d9084487b0e244c4fe. Selection source commit 891922bb1ec0b06924d210da680a8f56fb6bb324 records E5 revision f52bf8ec8c7124536f0efb74aca902b2995e5bcd and HyDE revision 7bcac572ce56db69c1ea7c8af255c5d7c9672fc2; validation metadata records reranker revision 233902d25c440f23af6f7d6e94d2946bac0bee0a.

## CoIR and FAISS retrieval paths

The saved cached-vector check under code-retrieval/artifacts/coir_official_cached_rev03/ reports CoIR nDCG@10 0.3259. cosqa.json SHA-256: e7d3a019ae856f707c7faf6e040448e9d237987783d1e764e07d9284674e56c8. metadata.json SHA-256: 880cd4596d2f2ab0e1c955d739c66de1f12e4c340f9cbec7e759a3b0770f3970. It records test qrels, 500 queries, 20,604 documents, CoIR.run, and normalized E5 vectors reused from cache d5127fd784781430; the model encoder was not rerun. This is a cached-vector path check, not a fresh official benchmark reproduction.

The local four-system result uses Faiss IndexFlatIP and the same CoIR metric evaluator. The local evaluator orders equal-score results by saved insertion order, then assigns unique ordinal scores. The installed coir-eval 0.7.0 exact-search source sorts corpus records by text length, uses torch.topk and a score/document-ID heap, and passes raw result scores to EvaluateRetrieval. Installed source hashes: exact_search.py ed32480e1ba7aff012b88d840e9cfc6ba2f8697fc52d4adc7d96b1961c5d8135; evaluation.py c22417b963105a2536d57ae3248bacc94ee993c000f4f70fb8b6c48bea3d2eac.

See the [CoIR dense exact-search source](https://github.com/CoIR-team/coir/blob/main/coir/beir/retrieval/search/dense/exact_search.py) and [CoIR evaluator source](https://github.com/CoIR-team/coir/blob/main/coir/beir/retrieval/evaluation.py). The saved evidence localizes the score discrepancy after encoding; it does not isolate a particular operation.

## Paper and submission identity

- Revision-28 anonymous review PDF and the root review PDF are byte-identical; both SHA-256 values are 0a0641f7a282f7aa73a7c05c2603d7121a73289a183225f6135fe5a807532ff9.
- code-retrieval/paper/main.pdf is author-identified and differs from the anonymous candidate.
- code-retrieval/paper/submission-anonymous.pdf is absent, although the README describes it as the intended EDAS upload copy.
- No EDAS receipt or exact upload was found among available paper/artifact files or audit attachments. The issue attachments are diagrams, not a manuscript.
- Historical revision notes describe anonymous PDFs as intended upload copies. They establish intent, not upload identity.

Verdict: revision 28 is the best local anonymous candidate, but the actual submission is unverified.

## Notebook and source provenance

The selection source commit 891922bb1ec0b06924d210da680a8f56fb6bb324 and final run commit 92c128695c2d5b2da7d1bd7e3d7d7feb53cfcc9b exist. The notebook at the final run commit loads the frozen selection and guards for valid selection, 500 queries, and depth 1,000. The final run records the same weights.

The run records notebook SHA-256 values db40bd4511055fefabe135dc968e87efbab6d63d8efec66466a4fc5450f7d4a9 and 68d39eb3408e003adb1663082f36609b8ac2efa5a165a42cfeb19a2a605f4fc6. The checked-out notebook hashes to ffb62b70d2160b3821be5962684dfe32a363ab68d21f3ad31439a86f12977763; its Git blob at the run commit hashes to aeef337445f08afab97c015a5200a1b02975469844e91751f81b465a66641822. The run directory has no executed notebook, and none of the available notebook files matches the recorded hashes. Exact executed notebook bytes remain unverified.

## Manual verification record

The issue commands were executed using Git Bash from Windows with project path /c/Users/Lyrid/Documents/Projects/research/code-retrieval. An initial WSL Bash invocation could not resolve that path; activation failed and global Python lacked numpy. That was failed setup, not a pass. The commands were rerun under Git Bash with the project environment. Steps 0–4 exited 0. The loader emitted an unauthenticated Hugging Face warning but completed.

Step 0:

    cd /c/Users/Lyrid/Documents/Projects/research/code-retrieval
    printf 'exit=%s\n' "$?"

Observed/pass: exit 0.

Step 1:

    source .venv/Scripts/activate
    printf 'exit=%s\n' "$?"

Observed/pass: exit 0; Python resolved to code-retrieval/.venv/Scripts/python.exe.

Step 2:

    python -c "from src.e5_baseline import BaselineConfig, load_cosqa; import json; v=load_cosqa(BaselineConfig(qrels_split='valid')); t=load_cosqa(BaselineConfig(qrels_split='test')); print(json.dumps({'valid_queries':len(v.queries),'test_queries':len(t.queries),'shared_query_ids':len(set(v.queries)&set(t.queries))}))"

Observed/pass, exit 0:

    {"valid_queries": 500, "test_queries": 500, "shared_query_ids": 0}

Step 3, run after environment activation. This reuses the existing loader, does not write files, and reports counts and hashes only:

    python - <<'PY'
    from collections import defaultdict
    from unicodedata import normalize
    import json
    from src.e5_baseline import BaselineConfig,load_cosqa,sha256_ids
    n=lambda s: " ".join(normalize("NFKC",str(s)).casefold().split())
    v=load_cosqa(BaselineConfig(qrels_split="valid")); t=load_cosqa(BaselineConfig(qrels_split="test"))
    def groups(d):
     g=defaultdict(list)
     for q,x in d.queries.items(): g[n(x)].append(q)
     return g
    def dup(g):
     d=[x for x in g.values() if len(x)>1]
     return [len(d),sum(map(len,d)),sum(len(x)-1 for x in d)]
    vg,tg=groups(v),groups(t); same=set(vg)&set(tg)
    vr={(n(v.queries[q]),d,s) for q,r in v.qrels.items() for d,s in r.items() if s>0}
    tr={(n(t.queries[q]),d,s) for q,r in t.qrels.items() for d,s in r.items() if s>0}
    vi={(q,d,s) for q,r in v.qrels.items() for d,s in r.items() if s>0}
    ti={(q,d,s) for q,r in t.qrels.items() for d,s in r.items() if s>0}
    print(json.dumps({"valid_queries":len(v.queries),"test_queries":len(t.queries),"valid_qrel_rows":sum(map(len,v.qrels.values())),"test_qrel_rows":sum(map(len,t.qrels.values())),"valid_query_id_sha256":sha256_ids(v.queries),"test_query_id_sha256":sha256_ids(t.queries),"valid_duplicate_groups_queries_excess":dup(vg),"test_duplicate_groups_queries_excess":dup(tg),"shared_query_ids":len(set(v.queries)&set(t.queries)),"shared_normalized_text_groups":len(same),"shared_positive_id_doc_score":len(vi&ti),"shared_positive_text_doc_score":len(vr&tr),"shared_relevant_docs":len({d for _,d,_ in vr}&{d for _,d,_ in tr})},sort_keys=True))
    PY
    status=$?
    printf 'STEP3 exit=%s\n' "$status"

Observed/pass, exit 0: 500 qrel rows in each split; both query-ID hashes above; duplicate group/query/excess counts 0/0/0 in each split; all cross-split overlap counts 0.

Step 4, run after environment activation:

    python - <<'PY'
    import hashlib,json
    from pathlib import Path
    def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    sp=Path("artifacts/e5_hyde_rerank/validation/weight_selection.json"); s=json.loads(sp.read_text())
    checks=[]
    for name,item in s["input_rankings"].items():
     p=Path(item["path"].replace(chr(92),"/")); checks.append(actual:= (p.exists() and sha(p)==item["sha256"]))
    val=Path("artifacts/e5_hyde_rerank/validation/rerank_depth1000/validation_results.json")
    r=json.loads(Path("artifacts/e5_hyde_rerank/runs/20260930T103232-1790764352158657000-80d721673d195f47-metrics-v1/result.json").read_text())
    print(json.dumps({"selection_sha256":sha(sp),"selection_split":s["selection_qrels_split"],"input_hash_matches":checks,"validation_hash_match":sha(val)==s["reranker_validation_summary"]["sha256"],"run_id":r["run_id"],"run_split":r["dataset"]["splits"]["qrels"],"weights_match":r["fusion"]["component_weights"]=={k:x["selected"]["weights"] for k,x in s["systems"].items()}},sort_keys=True))
    PY
    status=$?
    printf 'STEP4 exit=%s\n' "$status"

Observed/pass, exit 0: selection hash aef1dda81f7cd8c207e823dc50a4314ce73bffcffed9be8a26ac215f66a98ef2; all four input hashes matched; validation-summary hash matched; run split test; run weights matched.

The user waived Linear SVG browser-rendering confirmation. This issue and report make no in-app rendering claim. Ignored paper, cache, ranking, and run artifacts were preserved unchanged.
