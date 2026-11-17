---
name: main-result-figure
description: Use when planning experiments, designing the headline/main result figure of a paper, or checking whether planned experiments and logged data can produce that figure (including uncertainty).
---

Read `docs/guides/THE-MAIN-RESULT-FIGURE.md` and apply it. Produce:
1. **Takeaway**: one sentence a reader should leave the figure with.
2. **Mock figure**: axes, series, chart type, and the expected pattern. Use placeholder data if there are no results yet.
3. **Required data**: each field needed to draw it, and which experiment produces it. Flag any gaps.
4. **Storage format**: the tidy schema that would make plotting trivial (e.g. one row per run × condition × seed).
5. **Uncertainty**: where variance comes in (seeds, sampling, raters, prompts...) and whether the recorded data can measure it (CIs, error bars).

If the project has an abstract, check that the figure's takeaway matches its Findings.
