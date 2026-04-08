# DACR-Bench: Domain-Agnostic Causal Reasoning Benchmark

A static benchmark for evaluating multi-hop reasoning over real technical documents. Tests whether models can chain evidence across sections, compute derived values, and identify authoritative information when a document contains conflicting data.

## Quick Start

```bash
# Run a model against the benchmark
npx tsx src/run.ts \
  --benchmark data/dacr_bench_v1.0.json \
  --model sonnet

# Evaluate predictions
npx tsx src/evaluate.ts \
  --benchmark data/dacr_bench_v1.0.json \
  --predictions results/predictions_sonnet.json

# Quick test with mini subset (10 challenges)
npx tsx src/run.ts \
  --benchmark data/dacr_bench_v1.0_mini.json \
  --model sonnet --limit 5
```

## What This Benchmark Tests

Standard multi-hop QA benchmarks (HotpotQA, MuSiQue) use short Wikipedia passages, test short-answer extraction, and don't include conflicting information or numerical computation. DACR-Bench addresses these gaps.

Each challenge gives the model a **real technical document** — an actual arXiv paper or a procedurally generated technical document — along with **10 questions** that test 7 distinct reasoning skills:

### The 7 Question Categories

| Category | Hops | What It Tests | Example |
|----------|------|---------------|---------|
| **direct_extraction** | 1 | Single fact lookup | "What is the full name of HSSE?" → "hierarchical sheaf spectral embedding" |
| **multi_hop_bridge** | 2-3 | Connect facts via a bridge entity across sections | "What formula defines k(d), and how is the scale parameter eta defined?" → Must find formula in one section and eta's definition in another |
| **comparative** | 2 | Compare two entities on the same attribute | "What resolution do both UMAP and t-SNE rely on?" |
| **computation** | 2+ | Extract numbers and compute a derived value | "The framework has N stages and M steps. How many more steps than stages?" |
| **conditional_filtered** | 2+ | Apply a filter condition before answering | "What boundary condition applies specifically when k equals 0?" |
| **conflict_targeted** | 1-2 | Document contains conflicting values for the same fact; model must identify which is authoritative | "On how many datasets was HSSE evaluated?" → Document states both 12 (authoritative, in results) and 15 (conflicting, injected earlier). Model must reason about which is the actual reported result |
| **cross_section_synthesis** | 3+ | Integrate information scattered across 3+ sections | "What is the submission date, publication date, and subject category?" |

### Information Conflicts

Real technical documents frequently contain internal inconsistencies — preliminary estimates differ from final results, abstracts summarize imprecisely, different sections use different rounding. DACR-Bench systematically tests this by injecting conflicting values for specific facts at different document locations.

The model receives **no warning** about these conflicts. It must perform causal reasoning to determine which value is the consequence of the actual process described in the document (the experiment, the final analysis, the reported result) vs. which is a preliminary, estimated, or contextually subordinate claim.

This is measured as the **Causal Authority Resolution (CAR)** score — the fraction of conflict-targeted questions where the model identifies and uses the authoritative value.

### Example Challenge

**Document:** Real arXiv paper on Hierarchical Sheaf Spectral Embedding (11,280 words, biology)

**Questions:**
```
q01 [direct_extraction, 1 hop, easy]
    "What is the full name of HSSE?"
    → "hierarchical sheaf spectral embedding"

q04 [multi_hop_bridge, 2 hops, medium]
    "What formula defines k(d), and how is eta defined?"
    → "k(d) = exp(-d^2 / eta^2), where eta = median of nonzero
       pairwise distances on the vertex set"

q07 [computation, 2 hops, easy]
    "How many more steps than stages in the HSSE framework?"
    → "2"

q09 [conflict_targeted, 1 hop, easy]
    "On how many datasets was HSSE evaluated?"
    → "12"  (document also states "15" in a different location)

q10 [cross_section_synthesis, 3 hops, medium]
    "What is the submission date, publication date, and category?"
    → "submitted 2026-03-27, published 2026-03-31, cs.LG"
```

### Expected Model Output

```json
{
  "q01": {"answer": "hierarchical sheaf spectral embedding", "citation": "Title", "confidence": 0.95},
  "q04": {"answer": "k(d) = exp(-d^2/eta^2), eta is the median...", "citation": "Section 2.2", "confidence": 0.8},
  "q09": {"answer": "12", "citation": "Section 4, Table 1", "confidence": 0.85}
}
```

Models must provide a **confidence score** (0.0-1.0) per answer. The benchmark measures calibration — whether stated confidence predicts actual accuracy.

## Benchmark Composition

| Aspect | Value |
|--------|-------|
| Total challenges | 94 |
| Total questions | 940 |
| Information conflicts | 128 |
| Real-document challenges (arXiv) | 54 (57%) |
| Procedurally generated challenges | 40 (43%) |
| Domains | biology, chemistry, climate science, CS/AI, economics, materials science, medicine, NLP, physics, quantum physics, nuclear physics, computational biology, scRNA imputation |
| Difficulty | 35% easy, 43% medium, 22% hard |
| Hop depth | 35% single-hop, 38% 2-hop, 19% 3-hop, 8% 4+ hop |
| Mini subset | 10 challenges (1 per major domain), 100 questions |

### How DACR-Bench Compares

| Feature | MuSiQue | HotpotQA | BRIDGE | DocHop-QA | ConflictBank | **DACR-Bench** |
|---------|---------|----------|--------|-----------|-------------|----------------|
| Real documents | Wikipedia | Wikipedia | arXiv | PubMed | Wikidata | **arXiv (57%)** |
| Multi-hop (3+) | Yes | No | Yes | Yes | No | **Yes** |
| Numerical computation | No | No | No | No | No | **Yes** |
| Information conflicts | No | No | No | No | Inter-source | **Intra-document** |
| Citation grounding | No | Sentence | Evidence | No | No | **Yes** |
| Confidence calibration | No | No | No | No | No | **Yes** |
| Multi-domain | No | No | CS only | Biomed | Yes | **Yes (13)** |

## Evaluation Metrics

### Primary
- **Answer Accuracy (AA)** — fraction of questions answered correctly
- **Confidence-Weighted Accuracy** — accuracy weighted by model's stated confidence
- **Causal Authority Resolution (CAR)** — fraction of conflict-targeted questions where the model uses the authoritative value
- **Confidence Calibration (ECE)** — expected calibration error (lower = better)

### Secondary
- **Citation Grounding Score** — fraction of citations pointing to correct document location
- **Pass Rate** — fraction of challenges with AA >= 0.6

### Breakdowns
Results are broken down by question category (7), domain (13), hop depth (1-4+), and difficulty (easy/medium/hard).

## Running Models

### Claude CLI
```bash
npx tsx src/run.ts \
  --benchmark data/dacr_bench_v1.0.json \
  --model sonnet \
  --output results/predictions_sonnet.json
```

### Local Model via HTTP (vLLM, ollama, etc.)
```bash
npx tsx src/run.ts \
  --benchmark data/dacr_bench_v1.0.json \
  --runner http \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output results/predictions_qwen7b.json
```

### Python Runner
A Python runner is also available in the [evaluation results repo](https://github.com/botcoinmoney/synthetic-reasoning-transfer):

```bash
python run_benchmark.py \
  --benchmark data/dacr_bench_v1.0_mini.json \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output results/predictions.json
```

### Options
| Flag | Default | Description |
|------|---------|-------------|
| `--model` | sonnet | Model identifier |
| `--runner` | claude-cli | `claude-cli` or `http` |
| `--endpoint` | localhost:8000 | HTTP endpoint for local models |
| `--temperature` | 0.1 | Inference temperature |
| `--timeout` | 120 | Seconds per challenge |
| `--limit` | all | Max challenges to run |

## Evaluation

```bash
npx tsx src/evaluate.ts \
  --benchmark data/dacr_bench_v1.0.json \
  --predictions results/predictions_sonnet.json \
  --output results/results_sonnet.json
```

Produces `results_sonnet.json` (full scores) and `results_sonnet.md` (formatted report).

## Challenge Generation

Challenges were generated through two pipelines:

**Real-document challenges** are built from arXiv papers. The pipeline extracts structured facts from the paper, injects information conflicts by inserting plausible but incorrect values at different document locations, and generates questions across all 7 categories with automated multi-pass verification ensuring every question is answerable from the document.

**Engine-generated challenges** use procedural domain libraries to create synthetic documents with known ground-truth facts. Conflicts are injected deterministically, and solvability is verified algorithmically before inclusion.

All challenges undergo deterministic verification: every gold answer must be extractable from the document text, every conflict must present both the authoritative and conflicting values, and every multi-hop question must require the stated number of reasoning steps.

## File Structure

```
├── README.md
├── prompt_template.txt              # Evaluation prompt (no conflict warning)
├── data/
│   ├── dacr_bench_v1.0.json         # Full benchmark (94 challenges, frozen)
│   └── dacr_bench_v1.0_mini.json    # Mini subset (10 challenges)
└── src/
    ├── types.ts                     # Type definitions
    ├── run.ts                       # Model runner (Claude CLI + HTTP)
    └── evaluate.ts                  # Scoring + metrics
```

## Related

- **Fine-tuning results:** [botcoinmoney/dacr-bench-results](https://huggingface.co/datasets/botcoinmoney/dacr-bench-results) on HuggingFace
- **Training & evaluation code:** [botcoinmoney/synthetic-reasoning-transfer](https://github.com/botcoinmoney/synthetic-reasoning-transfer)
- **Training data:** [botcoinmoney/domain-agnostic-causal-reasoning-tuning](https://huggingface.co/datasets/botcoinmoney/domain-agnostic-causal-reasoning-tuning) on HuggingFace

## Citation

```bibtex
@misc{dacr_bench_2026,
  title={DACR-Bench: Domain-Agnostic Causal Reasoning Benchmark},
  author={botcoinmoney},
  year={2026},
  url={https://github.com/botcoinmoney/DACR-benchmark}
}
```
