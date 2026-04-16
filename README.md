<h1 align="center">LongCoT</h1>

<p align="center">Benchmarking Long-Horizon Chain-of-Thought Reasoning</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.14140"><img src="https://img.shields.io/badge/arXiv-2604.14140-b31b1b.svg" alt="arXiv"></a>
  <a href="https://longcot.ai"><img src="https://img.shields.io/badge/Website-longcot.ai-blue.svg" alt="Website"></a>
  <a href="https://huggingface.co/datasets/LongHorizonReasoning/LongCoT"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow.svg" alt="Dataset"></a>
</p>

<p align="center">
  Correspondence: <a href="mailto:sumeet.motwani@eng.ox.ac.uk">sumeet.motwani@eng.ox.ac.uk</a>, <a href="mailto:charles.london@cs.ox.ac.uk">charles.london@cs.ox.ac.uk</a>
</p>

LongCoT is a benchmark for measuring how well frontier LLMs can sustain coherent reasoning across extended chains of thought. Every problem pairs a **short input** with a **long reasoning output** (often spanning tens to hundreds of thousands of tokens) where individual steps are tractable in isolation but the difficulty emerges from *composition*: keeping plans on track, tracking state, propagating constraints, and recovering from mistakes without external tools or scaffolding.

The benchmark contains ~2,500 expert-designed problems across five domains (logic, computer science, chemistry, chess, mathematics), each with deterministic verification.

## Domains

| Domain | Questions | Verification |
|--------|-----------|--------------|
| `logic` | 500 | Programmatic (simulates/validates solution) |
| `cs` | 500 | JSON string match |
| `chemistry` | 500 | Regex-based SMILES extraction + canonical match (+ optional LLM fallback) |
| `chess` | 500 | Template-aware deterministic parsing (integer/FEN/SAN/move-dict) + engine checks |
| `math` | 500 | Deterministic math comparison (+ optional LLM fallback) |

Each domain has questions at three difficulty levels: `easy`, `medium`, `hard`. Two benchmarks are defined over this split:

- **LongCoT-Mini** — the `easy` subset (~500 questions), suitable for quick evaluation. Selected with `--difficulty longcot-mini`.
- **LongCoT** — `medium` + `hard` (~2,000 questions), the full benchmark. Selected with `--difficulty longcot`.

## Installation

LongCoT uses [uv](https://docs.astral.sh/uv/) for dependency management. Install uv if you don't already have it (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then:

```bash
uv sync
```

This creates a local `.venv`, installs LongCoT in editable mode, and pulls in all runtime dependencies (including the OpenAI, Anthropic, and Google GenAI SDKs used by the inference CLI) pinned to `uv.lock`. Pass `--no-dev` if you want to skip the test dependencies.

Run commands inside the environment with `uv run` (e.g. `uv run python run_inference.py ...`), or activate the venv manually with `source .venv/bin/activate`.

The math and chemistry fallback judges use Gemini and read `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

## Running inference

`run_inference.py` submits prompts to an LLM provider in parallel and writes one JSONL result per question to `responses/`.

```bash
# LongCoT-Mini (easy only) on GPT-5.2
uv run python run_inference.py --difficulty longcot-mini --config oai_gpt52

# Full LongCoT (medium + hard) on an OpenRouter model
uv run python run_inference.py --difficulty longcot --config or_deepseek

# A single domain, single difficulty
uv run python run_inference.py --domain math --difficulty hard --config anthropic_sonnet

# Sanity check without hitting the API
uv run python run_inference.py --config gemini_pro --dry-run --max-questions 5
```

Valid `--difficulty` values: `easy`, `medium`, `hard`, `longcot-mini` (alias for `easy`), `longcot` (alias for `medium` + `hard`). Omitting the flag runs all three difficulties.

### Configs

Configs are YAML files in `src/configs/`; pass the filename stem as `--config`. Built-in configs:

| Config | Provider | Model |
|--------|----------|-------|
| `oai_gpt52` | OpenAI | gpt-5.2 |
| `anthropic_sonnet` | Anthropic | claude-opus-4-6 |
| `gemini_pro` | Gemini | gemini-3-pro |
| `or_deepseek` | OpenRouter | deepseek-v3.2 |
| `or_glm` | OpenRouter | GLM |
| `or_grok` | OpenRouter | Grok |
| `or_kimi` | OpenRouter | Kimi |

API keys are declared in the configs with `${ENV_VAR}` placeholders (e.g. `${OPENAI_API_KEY}`) and resolved at runtime. Copy any of the existing YAML files to add a new provider/model.

### Output

Each response is saved as one JSON line with `question_id`, `successful`, `response_text`, `model`, `usage`, and (when the provider returns it) `reasoning`. Default output path is `responses/<domain>_<difficulty>_<model>_<timestamp>.jsonl`.

## Running analysis

`run_eval.py` takes a JSONL of responses, verifies each against the reference answers, and writes a results JSON to `results/`.

```bash
uv run python run_eval.py responses/math_hard_gpt-5.2_20260226.jsonl
```

Reported metrics:

| Metric | Description |
|--------|-------------|
| `correct` | Verified as correct |
| `incorrect` | Verified as wrong |
| `failed` | API error (no response returned) |
| `wrong_formatting` | Response did not include `solution = ...` (counted separately; verification is still attempted) |
| `accuracy` | `correct / (correct + incorrect)` — excludes failed calls |
| `overall_accuracy` | `correct / total` |

### LLM fallback judges

For math problems where deterministic comparison is inconclusive and for chemistry responses where SMILES extraction fails, a Gemini fallback judge is invoked. Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) to enable it. Pass `--no-fallback` to disable — scores may be lower without it. When fallback is enabled but no key is set, a warning is printed and those borderline cases fail closed.

## Python API

If you want to integrate LongCoT into your own harness rather than use the CLI:

```python
import longcot

# Load questions (optionally filter by domain / difficulty)
questions = longcot.load_questions(domain="logic", difficulty="easy")

# Each question is a dataclass with a uniform shape across domains
q = questions[0]
q.question_id   # "Sudoku_easy_1"
q.domain        # "logic"
q.difficulty    # "easy"
q.prompt        # Full prompt text to send to an LLM
q.problem       # dict (template metadata) or None
q.answer        # reference answer payload or None

# Call your LLM with q.prompt, then verify
correct = longcot.verify(q, response_text)
# -> bool

# Batch verification
results = longcot.verify_batch(questions, responses)
```

Fallback judges can be configured via `VerifyOptions`:

```python
options = longcot.VerifyOptions(
    math=longcot.MathVerifyOptions(enable_fallback=True),
    chemistry=longcot.ChemistryVerifyOptions(enable_fallback=True),
)
correct = longcot.verify(q, response_text, options=options)
```

All domains use `solution = ...` as the answer format; per-question prompts spell out the expected shape of the value.

## Submissions

We welcome community submissions to the LongCoT leaderboard.

To submit results:

1. Evaluate your model on the full benchmark using the default evaluation settings.
2. Include your model name, provider, and per-question outputs in a results file.
3. Open a pull request on the benchmark repository with the results and enough information for us to reproduce the run.

We verify submissions before adding them to the leaderboard.

## Authors

<p align="center">
  Sumeet Ramesh Motwani*, Daniel Nichols*, Charles London*, Peggy Li*, Fabio Pizzati*,<br>
  Acer Blake, Hasan Hammoud, Tavish McDonald, Akshat Naik, Alesia Ivanova, Vignesh Baskaran,<br>
  Ivan Laptev, Ruben Glatt, Tal Ben-Nun, Philip Torr, Natasha Jaques, Ameya Prabhu,<br>
  Brian Bartoldson, Bhavya Kailkhura, Christian Schroeder de Witt
</p>

<p align="center">
  University of Oxford · Lawrence Livermore National Laboratory · MBZUAI · KAUST ·<br>
  Hexo AI · University of Washington · University of Tübingen
</p>

<p align="center">
  *Joint first authors
</p>

## Citation

If you use LongCoT in your work, please cite:

```bibtex
@article{motwani2026longcot,
  title         = {LongCoT: Benchmarking Long-Horizon Chain-of-Thought Reasoning},
  author        = {Motwani, Sumeet Ramesh and Nichols, Daniel and London, Charles and Li, Peggy and Pizzati, Fabio and Blake, Acer and Hammoud, Hasan and McDonald, Tavish and Naik, Akshat and Ivanova, Alesia and Baskaran, Vignesh and Laptev, Ivan and Glatt, Ruben and Ben-Nun, Tal and Torr, Philip and Jaques, Natasha and Prabhu, Ameya and Bartoldson, Brian and Kailkhura, Bhavya and Schroeder de Witt, Christian},
  year          = {2026},
  eprint        = {2604.14140},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2604.14140}
}
```
