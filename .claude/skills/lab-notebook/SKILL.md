---
name: lab-notebook
description: Use after running an experiment, changing a model/pipeline, or making a research decision, to append a lab notebook entry; also when the user asks to log, record, or note what was done.
---

Append an entry following `docs/guides/LAB-NOTEBOOK.TEMPLATE.md` to the project's notebook. Use the existing notebook file if there is one. Otherwise ask the user where it should live.

Fill fields from evidence, not memory:
- **Date**: today
- **Researcher**: `git config user.name`
- **Git commit**: `git rev-parse HEAD`. Note it if there are uncommitted changes.
- **Background**, **Experiment/modification**, **Data used**, **Observations/results**: take these from the conversation and the actual outputs (logs, metrics). Include the exact numbers. Record surprises and unexplained effects as they happened.

Leave out any field you can't support with evidence. Write enough detail for someone else to replicate the step.
