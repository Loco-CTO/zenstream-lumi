# Quality evaluation and benchmarking

## Acceptance baseline

Use Qwen3.5 2B as the quality-acceptance baseline. Multilingual coverage must include Chinese; the maintained evaluation also covers Japanese. The local synthetic suite exercises follow-ups, recommendations, local grounding, web evidence fixtures, ambiguity, references, read-only limits, and unrelated requests.

The evaluator uses local synthetic fixtures and does not make web requests or download models. Outputs, model files, and evaluation reports belong in a user-selected data directory outside the repository. See `scripts/lumi_quality_eval.py` for the current scenario set and CLI options.

## Comparisons

- Record model revision and manifest, runtime release, quantization, host, context, selected cases, rubric, and judge.
- Use a judge independent of both models for a quality comparison. A model-assisted rubric score is not ground truth and still requires blind human review.
- Do not claim a clear win from incomplete language coverage, a small paired set, missing case scores, or an evaluation where the judge is one of the candidates.
- Inspect grounding, scope, safety, language, and category regressions as well as average score.

## Run the local evaluation

Use already installed models and write output outside Git:

```powershell
python scripts/lumi_quality_eval.py `
  --models-dir C:\path\to\lumi-models `
  --runtime-release-dir C:\path\to\runtime-release `
  --output-dir C:\path\to\lumi-quality-evals
```

For a paired model comparison, select an independent local judge and enough cases to cover English, Japanese, and Chinese. The script records provenance and case-selection hashes; review its generated report rather than treating aggregate scores alone as acceptance.

## Runtime benchmark

`scripts/benchmark_llama_cpp.py` measures the embedded runtime, not Orchestrator tools, web retrieval, HTTP/SSE delivery, or browser rendering. Keep the model and output paths outside the repository. See [runtime acceleration and streaming](runtime-acceleration-and-streaming.md) for modes, metrics, caveats, and the benchmark command.
