# DACR-Bench: Domain-Agnostic Causal Reasoning Benchmark

A static benchmark for evaluating whether LLMs can perform multi-hop reasoning over real technical documents. Tests if models trained on synthetic challenges generalize to real-world papers.

## Quick Start

```bash
# 1. Run a model against the benchmark
npx tsx src/run.ts \
  --benchmark data/dacr_bench_v1.1.json \
  --model sonnet

# 2. Evaluate predictions
npx tsx src/evaluate.ts \
  --benchmark data/dacr_bench_v1.1.json \
  --predictions results/predictions_sonnet.json

# Quick test with mini subset
npx tsx src/run.ts \
  --benchmark data/dacr_bench_v1.1_mini.json \
  --model sonnet --limit 5

# Run only real-document or synthetic challenges
npx tsx src/run.ts \
  --benchmark data/dacr_bench_v1.1.json \
  --model sonnet --split real

# Evaluate a specific split
npx tsx src/evaluate.ts \
  --benchmark data/dacr_bench_v1.1.json \
  --predictions results/predictions_sonnet.json \
  --split synthetic
```

## What's In The Benchmark

**Static dataset.** Same documents, questions, and gold answers every run. No regeneration.

| Component | v1.0 | v1.1 |
|-----------|------|------|
| Total challenges | 94 | 250 (target) |
| Total questions | 940 | 2500 (target) |
| Real-document challenges (arXiv) | 54 (57%) | 150 (60%) |
| Synthetic challenges (engine-generated) | 40 (43%) | 100 (40%) |
| Domains | 13 | 16+ |
| Max hop depth | 4 | 6+ |
| Independent split evaluation | No | Yes |

### Splits: Real vs Synthetic

v1.1 introduces **first-class split evaluation**. Each challenge is tagged `"real"` or `"synthetic"`:

- **Real** — challenges built from arXiv papers via LLM fact extraction
- **Synthetic** — challenges generated procedurally from domain libraries

The `--split` flag on both `run.ts` and `evaluate.ts` lets you test each independently. The evaluation report includes a **Transfer Gap** metric: `|AA_real - AA_synthetic|` — lower means synthetic performance predicts real-doc performance.

Split-specific dataset files are also generated:
- `dacr_bench_v1.1_real.json` — real-document challenges only
- `dacr_bench_v1.1_synthetic.json` — engine-generated challenges only
- `dacr_bench_v1.1_mini.json` — balanced mini subset (equal real + synthetic)

### Domains

**Real documents (arXiv):** biology, chemistry, climate science, CS/AI, economics, materials science, medicine, NLP, physics

**Synthetic (engine-generated):** quantum physics, nuclear physics, computational biology, scRNA imputation

### Question Categories

1. **direct_extraction** — Single fact lookup (1 hop)
2. **multi_hop_bridge** — Connect facts via bridge entity (2-3 hops)
3. **comparative** — Compare entities on same attribute (2 hops)
4. **computation** — Extract numbers and compute (2+ hops)
5. **conditional_filtered** — Apply filter before answering (2+ hops)
6. **trap_targeted** — Answer depends on conflicting info (model must identify authoritative source)
7. **cross_section_synthesis** — Integrate 3+ sections (3-6 hops)

### Hop Depth Distribution (v1.1 target)

| Hops | % | Description |
|------|---|-------------|
| 1 | 20% | Direct lookups |
| 2 | 25% | Pairwise reasoning |
| 3 | 25% | Three-step chains |
| 4 | 15% | Deep multi-hop |
| 5 | 10% | Extended cross-section synthesis |
| 6+ | 5% | Gauntlet questions |

## Model Output Format

The model must output JSON with **answer**, **citation**, and **confidence** per question:

```json
{
  "q01": {"answer": "73%", "citation": "Results section paragraph 14", "confidence": 0.95},
  "q02": {"answer": "35.5%", "citation": "Control viability in Methods, treated in Results", "confidence": 0.7}
}
```

**Confidence is required** (0.0 = pure guess, 1.0 = certain). The benchmark measures calibration — how well confidence predicts correctness.

**No conflicting-info warning.** The evaluation prompt does NOT tell models that documents may contain conflicting information. This tests raw reasoning ability.

## Running Models

### Claude CLI (default)
```bash
npx tsx src/run.ts \
  --benchmark data/dacr_bench_v1.1.json \
  --model sonnet \
  --output results/predictions_sonnet.json
```

### Local model via HTTP (vLLM, ollama, etc.)
```bash
npx tsx src/run.ts \
  --benchmark data/dacr_bench_v1.1.json \
  --runner http \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output results/predictions_qwen7b.json
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
| `--split` | all | `real`, `synthetic`, or `all` |

## Evaluation Metrics

### Primary
- **Answer Accuracy (AA)** — fraction correct across all questions
- **Confidence-Weighted Accuracy** — accuracy weighted by model's stated confidence
- **Confidence Calibration (ECE)** — expected calibration error (lower = better calibrated)

### Secondary
- **Trap Evasion Rate** — fraction of trap questions where model used authoritative value
- **Citation Grounding Score** — fraction of citations pointing to correct location
- **Pass Rate** — fraction of challenges with AA >= 0.6
- **Transfer Gap** — |AA_real - AA_synthetic| (v1.1+)

### Breakdowns
- By question category (7 types)
- By domain (13+ domains)
- By hop depth (1, 2, 3, 4, 5, 6+)
- By difficulty (easy, medium, hard)
- By split (real, synthetic)

## Evaluation Output

```bash
npx tsx src/evaluate.ts \
  --benchmark data/dacr_bench_v1.1.json \
  --predictions results/predictions_sonnet.json \
  --output results/results_sonnet.json
```

Produces:
- `results_sonnet.json` — full results with per-question scores
- `results_sonnet.md` — formatted markdown report table (includes split breakdown)


## File Structure

```
├── README.md
├── prompt_template.txt              # Evaluation prompt (no conflicting-info warning)
├── data/
│   ├── dacr_bench_v1.1.json         # Full benchmark (262 challenges, frozen)
│   ├── dacr_bench_v1.1_real.json    # Real-document split (102 challenges)
│   ├── dacr_bench_v1.1_synthetic.json # Engine-generated split (160 challenges)
│   ├── dacr_bench_v1.1_mini.json    # Balanced mini subset
│   ├── dacr_bench_v1.0.json         # v1.0 benchmark (94 challenges)
│   └── dacr_bench_v1.0_mini.json    # v1.0 mini subset
└── src/
    ├── types.ts                     # Type definitions
    ├── run.ts                       # Model runner (--split support)
    └── evaluate.ts                  # Scoring + metrics (--split support)
```

## Related

- **Fine-tuning results:** [botcoinmoney/dacr-bench-results](https://huggingface.co/datasets/botcoinmoney/dacr-bench-results) on HuggingFace
- **Training & evaluation code:** [botcoinmoney/synthetic-to-real-reasoning](https://github.com/botcoinmoney/synthetic-to-real-reasoning)
- **Training data:** [botcoinmoney/domain-agnostic-causal-reasoning-tuning](https://huggingface.co/datasets/botcoinmoney/domain-agnostic-causal-reasoning-tuning) on HuggingFace

## Citation

```bibtex
@misc{dacr_bench_2026,
  title={DACR-Bench: Domain-Agnostic Causal Reasoning Benchmark},
  author={botcoinmoney},
  year={2026},
  url={https://github.com/botcoinmoney/dacr-bench}
}
```
