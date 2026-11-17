## Preregistration: <project>

**Status:** DRAFT, <date>. **Freeze it by committing this file before the first real run.** Any later change goes in `LAB-NOTEBOOK.md` with a date and reason, and the old text stays: strike through it, don't delete it.

### 1. Primary quantity and evaluation
> The metric, exactly how it's computed, the evaluation mode (e.g. greedy vs sampled), n per condition, and how CIs are computed.

### 2. Setup
> Models, checkpoints, data (with versions and splits), and the config. Note anything changed from upstream defaults and why.

### 3. Hypotheses and unit of analysis
> Copy the prediction and falsifier from the hypothesis card. State what one data point is (a run, a seed, a model variant) and how many there are.

### 4. Analysis plan and statistics
> The tests you'll run, correction for multiple comparisons, and what counts as a detected effect.

### 5. Baselines
> Trivial baselines, an oracle or upper bound, and the strongest prior method.

### 6. Stop criteria
> When you stop, pivot, or scale up. Copy them from the hypothesis card.

### 7. Logged schema
> One row per (run × condition × seed × checkpoint): the columns you'll log. This should make the main figure trivial to plot (see `THE-MAIN-RESULT-FIGURE.md`).

### Amendments
> Changes made after this was frozen: date, what, why. Strike through the old text above instead of deleting it.
