# Random-init intent smoke

**Date:** 2026-10-03
**Evidence level:** 1 — exploratory
**Decision:** keep only as an engineering smoke baseline; reject it for product-quality use.

## Question

Can Lumi run a tiny ZenStream-specific intent and title-span predictor from random initialization through CPU training, objective improvement, artifact save/reload, structured inference, and basic resource measurement without pretrained assets?

## Data scope and integrity

- Source: examples supplied by the user in the Lumi goal attachment (`sha256:e6d3472313dd1f9dc3b80443ab6e90276245f3eb0f66f90519f91ff184750bc2`). No external dataset or model generated these examples.
- Allowed use recorded in the local manifest: this project's local exploratory training and development-only smoke evaluation. No release or distribution approval.
- Dataset: 12 training and 5 exploratory development rows; 8 train families and 4 disjoint development families. The tiny prompt-derived split is not a benchmark and has no qualified Japanese review.
- Dataset SHA-256: `7b547556886fcc8dbbd96f17da049bf6a03410f555c1be59cdb647feed57f4fd`.
- The runner is pinned to this exact dataset hash and a fixed source description. It rejects replacement or derived input instead of acting as a general training entry point; this also keeps the separately excluded 101-case draft and derivatives out of this smoke.
- The JSONL, source manifest, outputs, logs, and weights remain under local `.lumi-data` controlled storage outside every Git worktree. They are not part of the production data manifest or formal evaluation inventory.

## Candidate and training

- Candidate: two separate one-hidden-layer MLP heads over fixed, signed-hash Unicode character n-gram features (2,048 dimensions); 48 hidden units for intent classification and 32 for character-level BIO title tagging.
- Task-specific outputs: decision plus draft capability IDs (`playback.start`, `catalog.search`, `catalog.home.read`), with a bounded JSON-shaped response. The predictor proposes actions only; it does not call ZenStream APIs, negotiate playback, or execute state changes.
- Argument scope: the title head emits a title span; catalog search copies the original query as one string and does not extract genre, year, runtime, or watched-state constraints.
- Initialization: Gaussian random weights, seed 1729; no pretrained weights, checkpoints, embeddings, adapters, tokenizer model, teacher output, or GPU use.
- Recipe: Adam, learning rate 0.02, batch size 16, 120 epochs, one CPU thread. 164,264 total parameters.
- Training objective: intent cross-entropy 1.615589 → 0.00002314; slot cross-entropy 1.097658 → 0.130167. Training accuracy was 100% for intent and 92.74% for slot tags. This sharp train/dev gap is consistent with overfitting; it does not show useful generalization.
- Training time: 0.81 seconds on the recorded machine.

## Results

| Measure | Result |
|---|---:|
| Structured outputs passing the prototype schema check | 5/5 |
| Exact semantic matches | 2/5 (40%) |
| False actions | 2/5; both future-intent examples |
| Exact title span | 0/1 applicable dev examples |
| Model parameters | 164,264 |
| Compressed weights artifact | 618,501 bytes |
| Artifact SHA-256 | `e19200eae0b82da9237f08d5049e4f18fc43e345e1fe83abd8f3eb9ece75561e` |
| Artifact load | 17.43 ms |
| First post-load inference | 0.95 ms |
| Warm latency, 125 short calls | p50 0.40 ms; p95 0.92 ms |
| Process working set | 40.4 MB before reload; 41.6 MB after reload; 42.7 MB peak |

The working-set figures include Python and NumPy and are not model-only memory. Process startup, long-context prefill, production request distribution, background idle use, unload/wake behavior, and other CPU classes were not measured. The 5-row result is too small and unreviewed for a product-quality or target-threshold claim.

## Environment and reproducibility

- Windows x64; AMD processor family 25, model 97, stepping 2. A GeForce RTX 3070 was present but unused.
- Python 3.12.14, NumPy 2.3.5; OpenBLAS single-threaded.
- Runner source SHA-256: `98773fb5bf92cf9184f0bab5d0a4ab59eeef037d9f9f46aae30ce0a3764a7d61` (the `random_init_intent_smoke.py` source in this change).
- Base repository revision: `a76818468eeca7d53e656f88d1451f9c8c875c0e`.
- Full machine-readable report, per-epoch losses, model and metadata are retained locally alongside the dataset, not in Git.

With Python 3.12 and the pinned exploratory dependency installed, keep `$DataRoot` outside all repositories and use an empty/new output directory:

```powershell
python -m pip install -r experiments/requirements-exploratory.txt
$DataRoot = 'C:\path\outside\repositories\lumi-random-init-intent-smoke-v1'
python experiments/random_init_intent_smoke.py `
  --data "$DataRoot\examples.jsonl" `
  --output-dir "$DataRoot\rerun"
```

## Interpretation and next decision

This run demonstrates random initialization, real training, falling objective, saved artifact, reload, CPU inference, schema-shaped structured output, and basic latency/memory measurement. It is only an intent-component smoke: it emits no natural-language conversational response, and it does not evaluate greetings, acknowledgements, casual/media conversation, unsupported requests, or low-confidence recovery. The one `clarify` output is a classification result, not evidence of useful clarification wording. This partial result cannot satisfy Lumi's user-facing conversation requirement.

The smoke exposes a safety-relevant generalization failure: the small heads confuse future intentions with current actions and cannot reliably preserve a title across English/Japanese code-switching. Do not scale this candidate or use its development rows to tune another candidate. The next model-quality comparison needs a separately authored and reviewed development set, including qualified Japanese review, future/negated intent, mixed-language titles, actual constraint extraction, ordinary harmless conversation, useful clarification, uncertainty, and unsupported capability cases. Final holdout completion and release-weight terms remain later gates.
