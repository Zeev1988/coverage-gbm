---
name: eval-design-checklist
description: Use when designing, building, or auditing an evaluation or benchmark (especially LLM evals, LLM-as-judge setups, human rating studies, or taxonomies of model behaviors), before running the full eval or before submitting results.
---

Read `docs/guides/EVAL-DESIGN-CHECKLIST.md` and go through every item against the eval in question (code, prompts, data, writeup).

For each section, give **pass / fail / unknown** with evidence (file:line, a config value, or a quote from the writeup). Treat "unknown" as fail until someone checks it.

Finish with the 3 most damaging gaps, ranked by how badly each could invalidate the main claim, plus the cheapest fix for each (e.g. "run 3 prompt paraphrases on 50 items").

If you find a mistake that has shown up before (in the notebook, earlier reviews, or this conversation), *suggest* adding it to the "Mistakes we keep seeing" list in `docs/guides/EVAL-DESIGN-CHECKLIST.md`. Show the exact line and edit the guide only once the user approves.
