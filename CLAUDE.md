# Research sprint

This repo is a research sprint started from the team template. Skills are in `.claude/skills/`; each one points at a guide in `docs/guides/`.

## Stack
- **Python:** `uv` for everything: envs, deps, running code (`uv run`, `uv add`). Never use pip or conda.
- **Compute:** RunPod GPU pods.
  - **Launch and kill with `tools/rp.py`** (`uv run tools/rp.py` for usage). It adds the pod prefix and copies outputs off before every kill, and the cost guard depends on both.
  - Use the RunPod plugin (`runpod@runpod`: MCP plus skills such as `pod-doctor` and `cost-audit`) for everything else: GPU availability, templates, debugging a pod, serverless, billing questions.
  - LLM docs: https://docs.runpod.io/llms.txt
  - Keys live in `.env` (see `.env.example`), never in git.
- **Models/data:** Hugging Face Hub (`HUGGINGFACE_TOKEN`).
- **Tracking:** Weights & Biases (`WANDB_API_KEY`). Log every run; read runs with the `wandb` Python API.
- **RL/training:** Ray + RLinf when the sprint does embodied RL.
- **Writing:** the lab notebook is mirrored to Notion. Papers are in LaTeX/Overleaf.
- **Notion, Linear, HF, RunPod:** plugins enabled in `.claude/settings.json`.
- **Second opinion:** get a Codex (cross-model) review before big decisions: committing to a direction, freezing the prereg, claiming a result.

## Rules
- Before anything touches a paid GPU, it passes local self-checks: `py_compile`, a `__main__` self-check, a stub test.
- Never launch paid compute without the user's explicit OK.
- On every pod launch:
  - start `nohup uv run tools/rp.py watchdog <pod> <hours> &`, where hours = expected runtime × 1.5 (on macOS the timer pauses while the laptop sleeps)
  - then start `/loop 15m /cost-guard`
- Pods write small outputs (logs, metrics, eval CSVs) to `/root/out`, which the guard copies off before any kill. Jobs push checkpoints and other large artifacts to the HF Hub, never to `/root/out`.
- If a pod was launched some other way, its name must still start with `RUNPOD_POD_PREFIX-`, or the guard ignores it.
- Pod names get the `RUNPOD_POD_PREFIX` prefix (default: the repo directory name). Add `-keep` to a name to protect it from the guard.
- Every number gets uncertainty and a source. Log experiments and decisions with `lab-notebook` (`docs/research/LAB-NOTEBOOK.md`).
- Freeze `docs/research/PREREGISTRATION.md` by committing it before the first real run. Later changes go in the notebook.

## Sprint flow
| Phase | Skill |
|---|---|
| Choose and vet a topic, check novelty | `research-topic` |
| Concrete prediction, falsifier, stop criteria, then the prereg | `hypothesis-card` |
| After every experiment or decision | `lab-notebook` |
| Draft or check the abstract | `abstract-v0` |
| Design the headline figure and data schema | `main-result-figure` |
| Design or audit an eval | `eval-design-checklist` |
| Weekly update to the mentor | `sprint-update` |
| Review a writeup or draft | `sprint-review` |
| Naming, blog post, thread, press | `research-dissemination` |
| Whenever a pod is running | `cost-guard` |
