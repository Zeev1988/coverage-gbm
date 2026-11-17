# Research sprint template repo: design

## Intent
A GitHub template repo the team clones at the start of every **research sprint** (ML research, ICWM-style).
It ships the team's research skills, the tech stack and working rules in `CLAUDE.md`, and a RunPod cost guard.
It's the distilled form of what worked in `icwm/gwangju`, minus everything ICWM-specific.

## Goals
- A new sprint goes from clone to its first hypothesis card without any setup beyond `.env`.
- The sprint flow is encoded as skills: research topic → hypothesis card + preregistration → lab notebook → abstract → main figure → eval checklist → sprint update/review → dissemination.
- No GPU pod sits idle and billing without someone (Claude) noticing within 15 minutes.

## Non-goals
- Engineering/product repo scaffolding (CI, app frameworks, pre-commit).
- Paper-writing pipelines (`paper-*`, `citation-audit`, `ablation-planner`): later than a sprint; people install them globally if they want.
- A Python package/`pyproject.toml`: each sprint runs `uv init` for its own deps. `tools/rp.py` is stdlib-only and runs with `uv run`.

## Anti-goals
- Bloat: skills that are hundreds of lines, depend on Codex/MCP to work, or trigger only on Chinese phrases. Every skill stays in the house style (below).
- A cost guard that destroys work: killing a pod before its outputs are copied off.
- ICWM content leaking in (its notebook, prereg, hypotheses).

## Constraints
- House style for skills: frontmatter `name` + `description` ("Use when …"), a body of about 5–15 lines that points at a guide in `docs/guides/`. The guide holds the substance.
- `uv` for all Python. Secrets live only in `.env` (already gitignored). Ship `.env.example`.
- Pods are launched with `volumeInGb: 0`, so terminating a pod deletes everything on it.

## Approaches considered
For skills taken from outside the team's set:
- **A. Rewrite in house style. Chosen.** The set stays consistent, dependency-free and short. It costs a rewrite.
- B. Copy upstream skills as they are: less work, but they're large, their triggers are inconsistent, and they break without Codex MCP.
- C. Ship none and rely on global installs: smallest, but teammates don't get them.

For the cost guard's power to kill:
- Ask before every kill: safe, but an idle pod can bill all night.
- **Auto-kill when idle without a reason, after copying outputs off; ask when unsure. Chosen.**
- Fully automatic, killing anything idle: risks killing pods that are downloading or loading.

For where the guard's judgement runs: **Claude via `/loop 15m`, chosen**, because "why is it idle?" needs judgement (reading processes and logs). Its weakness is that the loop dies with the session. The existing fixed-hours `rp.py watchdog` stays as the backstop.

## What's in the repo
```
CLAUDE.md                     tech stack + working rules + sprint flow (which skill when)
README.md                     "start a sprint": use template → copy .env.example → /research-topic
.env.example                  RUNPOD_API_KEY=, HUGGINGFACE_TOKEN=, WANDB_API_KEY=, RUNPOD_IMAGE=, RUNPOD_POD_PREFIX=
.gitignore                    existing; add runs/ and cost-guard.log
tools/rp.py                   RunPod helper (from gwangju/.context/week0/rp.py)
.claude/settings.json         enables the team's plugins (below)
.claude/skills/
  research-topic/             NEW
  hypothesis-card/            from gwangju; also points to the preregistration template
  lab-notebook/               from gwangju, unchanged
  abstract-v0/                from gwangju, unchanged
  main-result-figure/         from gwangju, unchanged
  eval-design-checklist/      from gwangju; adds 2 mistakes (below). Drop the ICWM citation from the existing entry and keep the mistake
  sprint-update/              from gwangju, unchanged
  sprint-review/              from gwangju; also references REVIEWING-ACADEMIC-WORK.md
  research-dissemination/     from gwangju, unchanged
  cost-guard/                 NEW
docs/guides/                  all gwangju guides + image.png, plus:
  RESEARCH-TOPIC.md           NEW
  PREREGISTRATION.TEMPLATE.md NEW (structure taken from ICWM's PREREGISTRATION.md, emptied)
docs/research/
  LAB-NOTEBOOK.md             header only: title placeholder, format pointer, update cadence/where-posted placeholders
```
The source for copied files is `/Users/guynachshon/conductor/workspaces/icwm/gwangju/{.claude/skills,docs/guides}`.

## CLAUDE.md content
**Stack** (confirmed by the user):
- Python with `uv` for everything.
- RunPod GPU pods via `tools/rp.py`, with the key in `.env`.
- Hugging Face Hub for models and data (`HUGGINGFACE_TOKEN`).
- Ray + RLinf when the sprint does embodied RL. It was used in ICWM; listed as "when relevant".
- Notion mirrors the lab notebook.
- LaTeX/Overleaf for papers.
- Codex (the Codex plugin) for a cross-model review before big decisions: committing to a direction, freezing the prereg, claiming a result.
- Experiment tracking: Weights & Biases (`WANDB_API_KEY` in `.env.example`). Read runs with the `wandb` Python API.
- Notion, Linear, HF: via plugins enabled in `.claude/settings.json` (see Integrations).

**Rules** (my calls, taken from how ICWM was run):
- Run local self-checks (`py_compile`, a `__main__` self-check, a stub test) before anything touches a paid GPU.
- Never launch paid compute without the user's explicit OK.
- On every launch, also start `nohup uv run tools/rp.py watchdog <pod> <hours>`, where hours = expected runtime × 1.5, then start `/loop 15m /cost-guard`.
- Pods write small outputs (logs, metrics, eval CSVs) under `/root/out`, which the guard copies off. Large artifacts (checkpoints) get pushed to the HF Hub by the job itself and never go to `/root/out`. *(my call: rsyncing tens of GB to a laptop would make "copy failed, ask" the normal path)*
- Every number gets uncertainty and a source. Log experiments and decisions with `lab-notebook`.
- Freeze `PREREGISTRATION.md` by committing it before the first real run. Later changes go in the notebook.

**Sprint flow:** a short table that maps each phase to its skill.

## Integrations (Notion, Linear, HF, W&B, RunPod)
The template ships no skills for these. `.claude/settings.json` enables existing plugins from the official marketplace (`anthropics/claude-plugins-official`, declared in `extraKnownMarketplaces`):
`notion@claude-plugins-official`, `linear@claude-plugins-official`, `huggingface-skills@claude-plugins-official` (which includes `hf-cli`).
A teammate who opens the repo is prompted to install them, then logs in to Notion/Linear once. No `.mcp.json`: the plugins bring their own MCP connections.
- **W&B:** no plugin. A `CLAUDE.md` rule says to log runs to W&B and read them with the `wandb` Python API.
- **RunPod:** the official plugin `runpod@runpod` (marketplace `runpod/runpod-plugins-official`) for discovery, templates, debugging and billing. Launching and killing stay on `tools/rp.py`, because the guard relies on its prefix and copy-then-kill. Added at the user's request, 2026-11-15.

## New skill: `research-topic`
Trigger: choosing or vetting a research direction, before the hypothesis card.
The guide `RESEARCH-TOPIC.md` paraphrases Ofir Press's five tips in our own words, with a link (https://ofir.io/Tips-for-Finding-Research-Topics/):
1. Start from a problem the community cares about, not a method.
2. Iterate fast with simple solutions. Test the cheapest version first, with an oracle or upper bound, before building the complex one.
3. Write a paper people want to read: it teaches something about an existing system, improves a benchmark (on more than accuracy: train/inference cost, memory, latency), or introduces a benchmark.
4. Aim at what the field will care about about 9 months out. Check: are your citations from the last 1–2 years?
5. Keep it simple to describe, solve and run.

Then a **novelty check**, distilled from the upstream `novelty-check` and from how ICWM did it:
- Extract 3–5 core claims.
- Search arXiv, Semantic Scholar and OpenReview with at least 3 phrasings per claim.
- Read the closest papers in full text and verify every arXiv ID.
- Output: the closest prior work for each claim, the remaining gap, and a verdict (novel / incremental / done).
- Optionally, a Codex cross-review.

The skill outputs a short assessment (novelty, feasibility, impact, likely reviewer score) and hands off to `hypothesis-card`.

## `hypothesis-card` change
Add one line: after the card is filled in, draft `docs/research/PREREGISTRATION.md` from `PREREGISTRATION.TEMPLATE.md`. Template sections, from ICWM: primary quantity and evaluation, setup (model, data, config), hypotheses with their unit of analysis, analysis plan and stats, stop criteria, and a change-log rule (strike through, don't delete).

## `eval-design-checklist` change
Add two entries to "Mistakes we keep seeing", from upstream `experiment-audit`:
- Ground truth built from model outputs.
- Metrics normalized by the model's own score, or numbers that don't trace back to a results file.

## New skill: `cost-guard`
Trigger: a RunPod pod is running, the user asks about GPU spend or idle pods, or the 15-minute loop fires.

On each pass:
1. Run `uv run tools/rp.py audit` to get every pod's status, cost, uptime and GPU/CPU utilization.
2. A pod counts as **idle** when every GPU is below 10% utilization.
3. For an idle pod, SSH in and find out why. Check `ps`, `nvidia-smi`, and the tail of recent logs under `/root/out`. Look for downloads, installs, model loading, CPU-bound preprocessing, a finished job, a crash, or a hang. If the job logs to W&B, check whether the run is still logging (`wandb` API).
4. Decide:
   - **Legitimate reason** (setup, download, loading, high CPU from expected work): keep it and log the reason.
   - **No reason, and idle on the previous pass too** (≥30 minutes): rsync `/root/out` to `runs/<pod_id>/`. If the rsync succeeds, run `rp.py kill`. If it fails, don't kill; notify and ask.
   - **Unsure** (for example, unknown processes): notify and ask. Don't kill.
5. Append one line per pod to `cost-guard.log`: timestamp, pod, util, verdict, reason. The previous pass's verdict is read from this log, which is how "idle on 2 consecutive checks" is tracked.
6. On every kill or question, send a notification with Claude Code's built-in push notification (the agent's own, with no webhook). Decided by the user.
7. When no pods are left, stop the loop.

Scope: the guard acts only on pods whose name starts with the sprint's prefix: `RUNPOD_POD_PREFIX` in `.env`, defaulting to the repo directory name. `rp.py launch` adds the prefix itself, so it can't be forgotten. Other pods on the account are reported, never killed. This is my call: the RunPod account may be shared.
Exemption: a pod whose name contains `-keep` is reported, never killed.

## `tools/rp.py` changes
Ported from `gwangju/.context/week0/rp.py`, keeping its commands (`avail`, `launch`, `status`, `kill`, `watchdog`, `audit`). Changes:
- `.env` is read from the repo root.
- The image is a `--image` argument, or `RUNPOD_IMAGE` in `.env`, instead of a hardcoded value.
- The SSH pubkey path comes from `RUNPOD_PUBKEY`, defaulting to `~/.ssh/runpod_id.pub`.
- Add an `ssh <pod>` command that prints the `ssh root@ip -p port` line the guard needs.
- Fail with a clear message when the key or pubkey is missing.
- `launch` prepends `RUNPOD_POD_PREFIX` (default: the repo directory name) to the pod name.
- New `save <pod> [dest]`: rsync `/root/out` to `runs/<pod_id>/`, returning nonzero on failure. Both `kill --save` and `watchdog` use it, so every automatic kill is copy-then-kill and the copy-failed rule lives in one place. *(my call: the watchdog previously hard-killed, which broke the anti-goal)*

## Left to the builder
- Exact wording of the skills and guides, within house style.
- How `rp.py` arguments are parsed. Keep the stdlib approach.
- README length: short.
