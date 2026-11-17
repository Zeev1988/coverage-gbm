## Eval Design Checklist
#### Why a Checklist?
Most rejected benchmark papers fail on eval design, not on the idea. In the ICLR case study (see `REVIEWING-ACADEMIC-WORK.md`), most of the weak points were eval flaws: a single short prompt, equally weighted criteria, and rater assistance that "changed" ratings without showing they "improved".

Go through this list **before** running the full eval, while the problems are still cheap to fix.

---

### Validity — does it measure what you claim?
- Does the claim's scope match what the eval covers (users, tasks, languages, settings)?
- Are the prompts representative of real use? Avoid one hand-written prompt.
- Are the categories in your taxonomy distinct, and is each one meaningful on its own?

### Robustness
- **Prompt sensitivity**: do the results hold across ≥3 paraphrases or system prompts?
- **Sampling**: temperature and seeds are fixed and reported, and results are given across seeds.
- **Contamination**: could the items be in the training data? Check, or use held-out or newly written items.

### Scoring
- **LLM-as-judge**: check agreement with human labels on a subset and report it (e.g. Cohen's κ). Swap judge models to check for self-preference.
- **Aggregation**: don't average criteria of unequal importance without a stated reason. Report each criterion separately.
- **Human raters**: report inter-rater agreement. If raters had assistance, show that it made the ratings more *correct*, not just different.

### Statistics
- Sample size is justified (a power calculation, or at least CIs that are narrow enough to separate the models).
- Report uncertainty on every number (bootstrap CIs or std across seeds).
- Correct for multiple comparisons when comparing many models or categories.

### Baselines & comparisons
- Include trivial baselines (random, majority class, prompt-only).
- Compare named, current models that readers care about (see `MAKING-VIRAL-RESEARCH-GO-VIRAL.md`).
- Pin and report model versions and API dates.

### Release
- Release the items, the scoring code, and the raw model outputs.
- Write a datasheet: sourcing, licensing, known biases, intended use.

### Mistakes we keep seeing
*Add any mistake that shows up in more than one sprint here, with a one-line description and where it was seen.*

- **Calling a single-protocol study a "benchmark" / scoping claims beyond the settings tested.** Seen in: ICLR climate-LLM case study (REVIEWING-ACADEMIC-WORK.md).
- **Ground truth built from model outputs.** A model makes the "reference," and agreement with it gets reported as accuracy. Seen in: agent-run experiments.
- **Self-normalized or untraceable numbers.** A metric is divided by the model's own max, or a reported number has no results file behind it. Seen in: agent-run experiments.
