## Sprint Update | Template
#### Why?
Your mentor has about five minutes for your update. Make those five minutes count: lead with results and the decisions you need from them, not a diary of what you did.

Build it from your lab notebook entries since the last update. Don't write it from memory.

---

### @August 8, 2024 — [EXAMPLE UPDATE]
- **TL;DR** *(1–2 sentences)*: *Google Trends seasonality improves MAPE by 2.5 ± 0.4% (3 seeds), which meets the prediction on our hypothesis card. Convergence is 30 minutes slower.*
- **Results** *(numbers with uncertainty; link the figure)*:
  - *MAPE 11.2 ± 0.3 → 8.7 ± 0.4 on the 2023 held-out season ([fig](link))*
  - *The lagged-trends check shows the gain holds at a 2-week lag, so it's not leakage.*
- **Hypothesis status**: *supported / refuted / unclear, and what changed on the card*
- **Blockers**: *the slow convergence; we don't know the cause yet.*
- **Decisions needed** *(from the mentor)*: *expand to all regions now, or debug convergence first?*
- **Next**: *the 2–3 things for next week*
- **Commits**: *b676ff0*

---

Rules:
- Every result has uncertainty and a link to its source (a figure, a notebook entry, or a commit).
- Report negative and null results too. They're results.
- If there's nothing to decide, say "No decisions needed."

The cadence and where updates get posted are set per project. Record them once in the lab notebook header (e.g. `Updates: weekly, Fri, #proj-x`).
