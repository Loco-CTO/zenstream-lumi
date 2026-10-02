# Research references

**Last reviewed:** 2026-10-02. These sources inform design research only. None of the linked datasets, models, checkpoints, weights, or corpus text has been used to train Lumi.

## Scaling, data, and scratch-trained small models

- Google Research. 2026. [ATLAS: Practical scaling laws for multilingual models](https://research.google/blog/atlas-practical-scaling-laws-for-multilingual-models/); paper: [ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality](https://arxiv.org/abs/2510.22037). 774 multilingual runs; useful for hypotheses about language mixtures and compute allocation. Generic language-loss objectives and data do not determine Lumi's task optimum.
- MiniLingua authors. 2025. [MiniLingua: A Small Open-Source LLM for European Languages](https://arxiv.org/abs/2512.13298). Reports a 1B multilingual model trained from scratch; useful as a multilingual scratch-training case study, with no evidence for Japanese or Lumi behavior.
- Urbizu et al. 2025. [Sub-1B Language Models for Low-Resource Languages: Training Strategies and Insights for Basque](https://aclanthology.org/2025.mrl-main.35/). Useful as a caution about scratch training under constrained target-language data; its continual-pretraining method is not admissible for Lumi's runtime model.
- DataComp-LM authors. 2024. [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794). Controlled study of data filtering, deduplication, and mixing across 412M–7B models; useful for designing data-quality ablations, not an approval of its source data for Lumi.
- BabyLM authors. 2026. [BabyLM 4 guidelines](https://babylm.github.io/guidelines.html). Useful for data-efficient and multilingual small-model research; competition constraints and evaluation are not equivalent to ZenStream tasks.

## English/Japanese and code-switching

- Yoo et al. 2025. [Code-Switching Curriculum Learning for Multilingual Transfer in LLMs](https://aclanthology.org/2025.findings-acl.407/). Peer-reviewed Findings of ACL 2025; includes Japanese experiments but starts from pretrained models.
- Mohamed et al. 2026. [Lost in the Mix: Evaluating LLM Understanding of Code-Switched Text](https://aclanthology.org/2026.acl-long.2080/). ACL 2026; linguistically grounded code-switch evaluation design and mitigation study. Its benchmark is not Lumi's domain-specific EN/JA evaluation.
- Zhu et al. 2026. [How Can Synthetic Data Improve Multilingual Language Model Pretraining? A Data Quality Perspective](https://aclanthology.org/2026.acl-long.1002/). ACL 2026; data synthesis, filtering, and diversity-aware corpus scoring. Language coverage and tasks differ from Lumi.
- Schaffelder and Gatt. 2026. [Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning](https://aclanthology.org/2026.findings-acl.360/). Findings of ACL 2026; studies source diversity, distribution collapse, and safeguards in fine-tuning. Do not assume the findings directly predict scratch training.
- Xu et al. 2026. [GenesisFunc: Multi-Agent Data Generation for Accurate and Generalizable Function-Calling](https://aclanthology.org/2026.acl-long.1319/). ACL 2026; multi-stage synthetic function-calling data generation and quality checks. Its reported student fine-tunes an 8B model; its API-call setting is not Lumi's validated ZenStream boundary.
- NTT Technical Review. 2026. [Tsuzumi: Challenges in Developing a Sovereign Large Language Model](https://www.ntt-review.jp/archive/ntttechnical.php?contents=ntr202608fa2_s.html). First-party reporting on Japanese-focused tokenization and data quality; reported product outcomes need independent Lumi measurement.
- National Institute of Informatics. 2026. [LLM-jp-4 release and training-corpus announcement](https://www.nii.ac.jp/en/news/release/2026/0403.html). Japanese-from-scratch program and corpus-scale reference: it reports a 10.5T-token pretraining run and 1.2T-token mid-training. These scales are not a direct resource target for Lumi.
- Lee et al. 2026. [Equity with Efficiency: An Empirical Study of Tokenizers for Multilingual Large Language Models](https://arxiv.org/abs/2606.15044). Controlled tokenizer and 1.5B-model study on 11 Southeast Asian languages; token efficiency and downstream task effects merit testing at Lumi scale, but its language set excludes Japanese.

## Data provenance and candidate corpora

- [Common Pile v0.1 project](https://www.commonpile.org/); Kandpal et al. 2025, [paper](https://papers.neurips.cc/paper_files/paper/2025/file/52acc050138d6f40dad6f12f91a4ce22-Paper-Datasets_and_Benchmarks_Track.pdf); and [curator notes on its open-license policy](https://huggingface.co/blog/stellaathena/common-pile). A substantial public-domain/openly licensed candidate with source-level curation, while its own materials document the difficulty of automated rights signals and public-domain proof. Audit constituent sources; do not treat the aggregate release as Lumi approval.
- Palen-Michel et al. 2022. [Multilingual Open Text Release 1](https://aclanthology.org/2022.lrec-1.224/). VOA news corpus reporting public-domain source material and CC BY 4.0 for the collection; includes English but not Japanese.
- [Aozora Bunko file handling rules](https://www.aozora.gr.jp/guide/kijyunn.html). The rules distinguish expired-copyright works from protected works and request retention of work and transcription provenance; check each work and translator separately.
- Japan Digital Agency. [Public Data License v1.0](https://www.digital.go.jp/en/resources/open_data/public_data_license_v1.0) and [Japanese-law QA announcement](https://www.digital.go.jp/news/9d06cea7-4b62-4d27-b036-d39cf65dad73). PDL permits use of covered content including commercial use with attribution, subject to exclusions and third-party rights; the released QA data is domain-specific and public, so it is not a sealed holdout.
- [NICT ParaNatCom English-Japanese abstract corpus](https://att-astrec.nict.go.jp/member/mutiyama/paranatcom/index.html). Its release states CC BY 4.0 and identifies the Nature Communications / PubMed lineage; useful only as narrow academic parallel text.
- [NICT Asian Language Treebank Parallel Corpus](https://att-astrec.nict.go.jp/member/mutiyama/ALT/index.html). The project identifies Wikinews source text, NICT translations, CC BY 4.0 corpus terms, and separate train/dev/test splits; use the training split only if later approved.
- [J-CHAT dataset card](https://huggingface.co/datasets/sarulab-speech/J-CHAT) and [paper](https://arxiv.org/abs/2407.15828). The current card limits use to a Japanese Copyright Act Article 30-4 purpose and states that commercial use is not admitted.
- [FineWeb2 dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), [processing repository](https://github.com/huggingface/fineweb-2), and [Common Crawl overview](https://commoncrawl.org/overview). Useful language-processing research, but its collection-level ODC-By terms do not by themselves establish rights to each crawled web page.

## Sequence architectures

- Peng et al. 2025. [RWKV-7 "Goose" with Expressive Dynamic State Evolution](https://arxiv.org/abs/2503.14456). Reports multilingual models from 0.19B to 2.9B and recurrent-state inference; validate behavior, CPU performance, and implementation burden at Lumi sizes.
- Waleffe et al. 2024. [An Empirical Study of Mamba-based Language Models](https://arxiv.org/abs/2406.07887). Controlled 8B model comparison including Mamba-2, hybrid, and Transformer; hybrid and pure-SSM strengths differ by task. Scale and training budget are much larger than Lumi's likely early experiments.
- Yang et al. 2025. [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464), ICLR 2025; [official implementation](https://github.com/NVlabs/GatedDeltaNet). Candidate for small hybrid experiments; kernel and CPU portability are open questions.
- Dao and Gu. 2024. [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060). Introduces Mamba-2/SSD; useful architectural background, not direct evidence of a Lumi win.

## Structured generation and reliability

- Geng et al. 2025. [Generating Structured Outputs from Language Models: Benchmark and Studies](https://arxiv.org/abs/2501.10868); [JSONSchemaBench](https://github.com/guidance-ai/jsonschemabench). Evaluates structured decoding validity, schema coverage, efficiency, and output quality; syntax and semantic correctness must be scored separately.

## Inference, quantization, and deployment documentation

- [llama.cpp official repository and supported backends](https://github.com/ggml-org/llama.cpp). Current cross-platform inference and quantization candidate; support for a future Lumi architecture is not yet established.
- [ONNX Runtime GenAI installation and execution providers](https://onnxruntime.ai/docs/genai/howto/install.html). Documents separate CPU, CUDA, and DirectML packages; use as a portability candidate, not a selected runtime.
- [TorchAO quantized inference workflows](https://docs.pytorch.org/ao/stable/workflows/inference.html). Current quantization options include CPU-oriented x86 and accelerator paths; hardware-specific support and accuracy must be measured.
- Xia et al. 2026. [Efficient INT8 Inference of Small NLP Models on Server CPUs with PyTorch Native Stack](https://arxiv.org/abs/2608.18182). Reports CPU INT8 results for BERT-family encoders on Xeon; adjacent evidence only, not autoregressive Lumi benchmark evidence.
