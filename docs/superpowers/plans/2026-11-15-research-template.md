# Research Sprint Template Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn this empty repo into the team's research-sprint template: 10 skills plus their guides, a RunPod helper with a cost guard, `CLAUDE.md`, and plugin config.

**Architecture:** Each skill is a short `SKILL.md` that points at a guide in `docs/guides/`. `tools/rp.py` is a stdlib-only RunPod GraphQL CLI, and the `cost-guard` skill drives it from a `/loop 15m`. Integrations come in through `.claude/settings.json`, which enables plugins from the official marketplace.

**Tech Stack:** Markdown, JSON, Python 3 stdlib (run with `uv run`), RunPod GraphQL, rsync/ssh.

**Spec:** `docs/superpowers/specs/2026-11-15-research-template-design.md`

**Source:** `SRC=/Users/guynachshon/conductor/workspaces/icwm/gwangju`

## Global Constraints
- Skill house style: frontmatter has `name` (equal to the directory name) and a `description` starting with "Use when". The body is about 5–15 lines and points at a `docs/guides/` file.
- No ICWM content in the template: no "Ranking Is Not Training", no ICWM hypotheses or numbers. RLinf is mentioned only in `CLAUDE.md`'s stack list, as "when relevant".
- `uv` for all Python. `tools/rp.py` stays stdlib-only, with no `pyproject.toml`.
- Secrets only in `.env`. Ship `.env.example` with `RUNPOD_API_KEY=`, `HUGGINGFACE_TOKEN=`, `WANDB_API_KEY=`, `RUNPOD_IMAGE=`, `RUNPOD_POD_PREFIX=`.
- Pods have `volumeInGb: 0`. Every automatic kill is copy-then-kill: if the copy fails, the pod is not killed.
- Idle means every GPU below 10% utilization. Kill after 2 consecutive idle passes (≥30 minutes) with no legitimate reason.
- The guard acts only on pods whose name starts with `RUNPOD_POD_PREFIX` (default: the repo directory name). It never kills a pod whose name contains `-keep`.
- Notifications use Claude Code's built-in push notification only.

## Review Focus
- **The prefix is a prefix of another sprint's prefix** (`icwm` vs `icwm2`): a pod is "ours" only if its name starts with `<prefix>-`, not just `<prefix>`. Pinned in the Task 3 selftest.
- **A pod is running but has no public SSH port yet** (still booting): `ssh_target` returns `None`. `save` returns `False` rather than crashing, so nothing is killed. Pinned in Task 3.
- **`/root/out` doesn't exist on the pod:** rsync exits nonzero, so the pod isn't killed and the guard asks. Covered by the `save` → `False` path in Task 3. The cost-guard skill text in Task 4 names this case.
- **`launch` called with a name that already has the prefix:** it must not become `p-p-name`. Pinned in Task 3.
- **`.env` missing or without `RUNPOD_API_KEY`:** exit with a clear message, not a `KeyError`. Pinned in Task 3.

---

### Task 1: Port the team's 8 skills and guides

**Files:**
- Create: `.claude/skills/{abstract-v0,eval-design-checklist,hypothesis-card,lab-notebook,main-result-figure,research-dissemination,sprint-review,sprint-update}/SKILL.md` (copied from `$SRC/.claude/skills/`)
- Create: `docs/guides/*` (copied from `$SRC/docs/guides/`, including `image.png`)
- Create: `docs/guides/PREREGISTRATION.TEMPLATE.md`
- Create: `docs/research/LAB-NOTEBOOK.md`
- Modify (after copying): `hypothesis-card/SKILL.md`, `sprint-review/SKILL.md`, `docs/guides/EVAL-DESIGN-CHECKLIST.md`

**Interfaces:**
- Produces: the guide paths the skills reference. Task 5's `CLAUDE.md` flow table names these 8 skills.

- [ ] **Step 1: Copy** with `cp -R $SRC/.claude/skills/* .claude/skills/` and `cp $SRC/docs/guides/* docs/guides/`.
- [ ] **Step 2: Edit `hypothesis-card/SKILL.md`.** Add one bullet: after the card is filled in, draft `docs/research/PREREGISTRATION.md` from `docs/guides/PREREGISTRATION.TEMPLATE.md`. It gets frozen by being committed before the first real run.
- [ ] **Step 3: Edit `sprint-review/SKILL.md`.** In the calibration line, add: see `docs/guides/REVIEWING-ACADEMIC-WORK.md` for a worked case.
- [ ] **Step 4: Edit `EVAL-DESIGN-CHECKLIST.md`.**
  - In the existing "Mistakes we keep seeing" entry, delete `, ICWM27 "Ranking Is Not Training" abstract v0.1 (2026-11-14)`.
  - Append two entries:
    - **Ground truth built from model outputs** (a reference made by a model, then agreement with it reported as accuracy). Seen in: agent-run experiments.
    - **Self-normalized or untraceable numbers** (a metric divided by the model's own max, or a number with no results file behind it). Seen in: agent-run experiments.
- [ ] **Step 5: Write `PREREGISTRATION.TEMPLATE.md`.**
  - Structure: `## Preregistration: <project>`, then a status line with the freeze rule (commit before the first real run; later changes go in `LAB-NOTEBOOK.md` with date and reason; strike through old text, don't delete it).
  - Sections, each with a one-line prompt: 1. Primary quantity and evaluation (metric, eval mode, n, CIs); 2. Setup (models, data, config); 3. Hypotheses and unit of analysis; 4. Analysis plan and statistics; 5. Baselines; 6. Stop criteria.
  - Take the shape from `$SRC/docs/research/PREREGISTRATION.md`, with none of its content.
- [ ] **Step 6: Write `docs/research/LAB-NOTEBOOK.md`.**
  - Contents: `## Lab Notebook: <project>`, then `Format: docs/guides/LAB-NOTEBOOK.TEMPLATE.md. Newest entries at the bottom.`
  - Then `Update cadence: <ask>` and `Updates posted to: <ask>` (the `sprint-update` skill fills these in), then `---`.
- [ ] **Step 7: Verify.**
  - Run: `grep -rn -i "ranking is not training\|icwm" .claude docs/guides docs/research` → expect no output.
  - Run: `ls docs/guides | wc -l` → expect 11 (9 copied guides + image + prereg template).
- [ ] **Step 8: Commit:** `git add .claude docs/guides docs/research && git commit -m "feat: port team research skills and guides"`

### Task 2: `research-topic` skill

**Files:**
- Create: `.claude/skills/research-topic/SKILL.md`
- Create: `docs/guides/RESEARCH-TOPIC.md`

- [ ] **Step 1: Write the guide.**
  - **Part 1:** Ofir Press's five tips, paraphrased in our own words (don't copy the post's text), linking https://ofir.io/Tips-for-Finding-Research-Topics/. Use the five points listed in the spec's `research-topic` section, including the oracle/upper-bound test and the "citations from the last 1–2 years" check.
  - **Part 2:** the novelty check:
    - extract 3–5 core claims
    - search arXiv, Semantic Scholar and OpenReview with ≥3 phrasings per claim
    - read the closest papers in full text and verify every arXiv ID
    - for each claim, record the closest prior work and the remaining gap, then give a verdict: novel / incremental / done
    - optionally, a Codex cross-review
  - **Output format:** novelty, feasibility, impact, likely reviewer score (/10), and the narrowest version worth a hypothesis card.
- [ ] **Step 2: Write the skill.**
  - Description: `Use when choosing, vetting, or comparing research directions or topics, or checking whether an idea has already been done, before writing a hypothesis card.`
  - Body: read the guide; apply Part 1 as a critique of the idea, then run Part 2. Never cite a paper whose ID you haven't fetched. Next step: `hypothesis-card`.
- [ ] **Step 3: Verify.** `head -4 .claude/skills/research-topic/SKILL.md` shows `name: research-topic` and the description.
- [ ] **Step 4: Commit:** `git commit -m "feat: add research-topic skill"`

### Task 3: `tools/rp.py`

**Files:**
- Create: `tools/rp.py` (port of `$SRC/.context/week0/rp.py`; keep `gql`, `avail`, `status`, `audit` as they are)

**Interfaces (Produces, used by Task 4):**
- `uv run tools/rp.py audit`: same output as the source, plus `ours=yes|no` on each pod line
- `uv run tools/rp.py ssh <pod_id>`: prints `ssh root@<ip> -p <port> -i <key>`, or exits 1 if there's no public SSH port
- `uv run tools/rp.py save <pod_id> [dest]`: rsyncs `/root/out/` to `dest` (default `runs/<pod_id>/`); exits 0 on success, 1 on failure
- `uv run tools/rp.py kill <pod_id> [--save]`: with `--save`, terminates only if `save` succeeded, otherwise exits 1 without terminating
- `uv run tools/rp.py watchdog <pod_id> <hours>`: sleeps, then does `kill --save`
- `uv run tools/rp.py launch <name> <gpu> <count> <disk> [secure|community] [--image IMG]`
- `uv run tools/rp.py selftest`

Python functions (module level):
- `load_env(root: Path) -> dict[str, str]`: `.env` at `root`, with `os.environ` taking precedence
- `pod_prefix(env: dict, root: Path) -> str`: `env["RUNPOD_POD_PREFIX"]`, or `root.name` if missing or empty
- `prefixed(name: str, prefix: str) -> str`: `f"{prefix}-{name}"` unless `name` already starts with `f"{prefix}-"`
- `is_ours(pod_name: str, prefix: str) -> bool`: `pod_name.startswith(f"{prefix}-")`
- `ssh_target(pod: dict) -> tuple[str, int] | None`: `(ip, publicPort)` for the public port with `privatePort == 22`, else `None`
- `save(pod_id: str, dest: Path | None = None) -> bool`
- `kill(pod_id: str, save_first: bool = False) -> bool`: returns whether the pod was terminated

The repo root is `Path(__file__).resolve().parents[1]`. The key and pubkey are loaded lazily on first use, not at import time, so `selftest` runs without `.env`. `RUNPOD_IMAGE` comes from `--image`, then the env, then an error. `RUNPOD_PUBKEY` defaults to `~/.ssh/runpod_id.pub`; the private key is the same path without `.pub`. When `RUNPOD_API_KEY` is missing, exit with `missing RUNPOD_API_KEY: copy .env.example to .env`.

- [ ] **Step 1: Write the `selftest` command first.** It's plain asserts, with `gql` and `save` monkeypatched as module globals:

```python
assert pod_prefix({}, Path("/x/icwm")) == "icwm"
assert pod_prefix({"RUNPOD_POD_PREFIX": "s1"}, Path("/x/icwm")) == "s1"
assert prefixed("run", "s1") == "s1-run" and prefixed("s1-run", "s1") == "s1-run"
assert is_ours("s1-run", "s1") and not is_ours("s12-run", "s1") and not is_ours("s1", "s1")
pod = {"runtime": {"ports": [{"ip": "1.2.3.4", "isIpPublic": True, "privatePort": 22, "publicPort": 4022}]}}
assert ssh_target(pod) == ("1.2.3.4", 4022)
assert ssh_target({"runtime": None}) is None
# kill --save: failed save means no terminate
calls = []; globals()["save"] = lambda *a, **k: False; globals()["gql"] = lambda *a, **k: calls.append(a)
assert kill("p", save_first=True) is False and calls == []
globals()["save"] = lambda *a, **k: True
assert kill("p", save_first=True) is True and len(calls) == 1
print("selftest ok")
```

- [ ] **Step 2: Run it before implementing.** `uv run tools/rp.py selftest` → expect a `NameError` or `AttributeError`.
- [ ] **Step 3: Implement** the functions and commands listed in Interfaces.
  - `save` runs `rsync -az -e "ssh -p PORT -i KEY -o StrictHostKeyChecking=accept-new" root@IP:/root/out/ DEST/`. It returns `False` when `ssh_target` is `None` or rsync exits nonzero.
  - `ssh_target` must tolerate `runtime` being `None` or `ports` being `None`.
- [ ] **Step 4: Run the selftest.**
  - `uv run tools/rp.py selftest` → `selftest ok`.
  - `env -u RUNPOD_API_KEY uv run tools/rp.py audit` (with no `.env`) → exits with the missing-key message.
- [ ] **Step 5: Commit:** `git add tools/rp.py && git commit -m "feat: add RunPod helper with copy-then-kill"`

### Task 4: `cost-guard` skill

**Files:**
- Create: `.claude/skills/cost-guard/SKILL.md`
- Create: `docs/guides/COST-GUARD.md`

**Interfaces:**
- Consumes: the Task 3 commands `audit`, `ssh`, `save`, `kill --save`.
- Produces: `cost-guard.log` at the repo root (gitignored in Task 5). One line per pod per pass, tab-separated: `ISO-timestamp  pod_id  name  gpu_util  verdict  reason`, where verdict is `ok|keep|idle-1|killed|ask|not-ours`.

- [ ] **Step 1: Write the guide.**
  - Write the pass procedure from the spec's cost-guard section, steps 1–7, using the exact commands from Task 3.
  - **Idle rule:** every GPU below 10%.
  - **Previous verdict:** read from the last `cost-guard.log` line for that pod. `idle-1` followed by idle again with no reason means kill.
  - **Legitimate reasons to keep:** download, install, model load, expected CPU-bound work (high CPU), a W&B run still logging.
  - **Kill path:** only `uv run tools/rp.py kill <id> --save`. If it exits 1, the verdict is `ask`.
  - **Ask cases:** unknown processes, an SSH failure, `/root/out` missing.
  - **Never kill** pods that are `not-ours` or have `-keep` in their name. Report them.
  - **Notify:** a Claude Code push notification on every `killed` or `ask`.
  - **Stop:** stop the loop when the audit shows no running pods that are ours.
- [ ] **Step 2: Write the skill.**
  - Description: `Use when a RunPod pod is running, when launching one, when the 15-minute cost-guard loop fires, or when the user asks about GPU spend or idle pods.`
  - Body: if no loop is running, start one with `/loop 15m /cost-guard`. Otherwise run one pass per `docs/guides/COST-GUARD.md`. Copy-then-kill only; when unsure, ask.
- [ ] **Step 3: Verify.** `grep -c "kill <id> --save\|-keep\|10%" docs/guides/COST-GUARD.md` → expect ≥3.
- [ ] **Step 4: Commit:** `git commit -m "feat: add cost-guard skill"`

### Task 5: `CLAUDE.md`, README, config

**Files:**
- Create: `CLAUDE.md`, `README.md`, `.env.example`, `.claude/settings.json`
- Modify: `.gitignore` (append `runs/` and `cost-guard.log`; `.env` is already ignored at line 153)

- [ ] **Step 1: Write `CLAUDE.md`.**
  - **Stack:** the bullets from the spec's "CLAUDE.md content", including W&B and the plugins line.
  - **Rules:** the spec's rule bullets verbatim: self-checks before paid GPU, no launch without OK, watchdog with hours = expected runtime × 1.5 plus `/loop 15m /cost-guard`, `/root/out` small and checkpoints to HF Hub, uncertainty and source on every number, the prereg freeze.
  - **Sprint flow table:** research-topic → hypothesis-card → lab-notebook → abstract-v0 → main-result-figure → eval-design-checklist → sprint-update / sprint-review → research-dissemination, plus cost-guard "whenever a pod runs".
  - Keep it to about 60 lines or fewer.
- [ ] **Step 2: Write `README.md`.**
  - Steps: "Use this template" on GitHub → `cp .env.example .env` and fill it in → open in Claude Code and accept the plugin prompt → log in to Notion and Linear → run `/research-topic`.
  - Keep it to about 25 lines or fewer.
- [ ] **Step 3: Write `.env.example`** with the 5 keys from Global Constraints, each with a one-line comment.
- [ ] **Step 4: Write `.claude/settings.json`:**

```json
{
  "extraKnownMarketplaces": {
    "claude-plugins-official": {"source": {"source": "github", "repo": "anthropics/claude-plugins-official"}}
  },
  "enabledPlugins": {
    "notion@claude-plugins-official": true,
    "linear@claude-plugins-official": true,
    "huggingface-skills@claude-plugins-official": true
  }
}
```

- [ ] **Step 5: Verify.**
  - Run: `python3 -m json.tool .claude/settings.json >/dev/null && echo ok` → `ok`.
  - Run the house-style check → expect no output:

```bash
for d in .claude/skills/*/; do n=$(basename $d); f=$d/SKILL.md
  grep -q "^name: $n$" $f || echo "BAD name $n"
  grep -q "^description: Use when" $f || echo "BAD desc $n"
  [ $(awk '/^---$/{c++;next} c>=2' $f | wc -l) -le 20 ] || echo "LONG $n"
  grep -q "$n" CLAUDE.md || echo "NOT IN CLAUDE.md $n"
done
```

  - Run: `ls .claude/skills | wc -l` → `10`.
- [ ] **Step 6: Commit:** `git add -A && git commit -m "feat: add CLAUDE.md, README, env example, plugin config"`
