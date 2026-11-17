# Research sprint template

A starting point for a research sprint. It comes with the team's research skills for Claude Code, our stack and working rules in `CLAUDE.md`, and a RunPod cost guard.

## Start a sprint
1. On GitHub: **Use this template** → new repo, then clone it.
2. `cp .env.example .env` and fill it in.
3. Set `RUNPOD_POD_PREFIX` to something unique for this sprint: the cost guard only touches pods with that prefix.
4. Open the repo in Claude Code and accept the prompt to install the team plugins (Notion, Linear, Hugging Face, RunPod). Sign in to Notion, Linear and RunPod once (`/mcp`).
5. Run `/research-topic` with your idea.

The flow from there (hypothesis card → notebook → abstract → figure → updates) is in `CLAUDE.md`.

## What's here
- `.claude/skills/`: 10 short skills, each pointing at a guide in `docs/guides/`
- `docs/research/`: your lab notebook and preregistration
- `tools/rp.py`: the RunPod helper (`uv run tools/rp.py` for usage, `uv run tools/rp.py selftest` to check it)
