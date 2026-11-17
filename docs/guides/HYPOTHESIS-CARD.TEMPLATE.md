## Hypothesis Card | Template
#### Why Write One Before You Start?
*If you can't say in advance what result would prove you wrong, you can't learn from the result.*

Writing the hypothesis down before running anything:
- Stops you from moving the goalposts once results come in
- Forces you to choose a measurable prediction instead of a vague direction
- Gives your sprint review something concrete to check the outcome against

Like the Abstract V0.x, it's fine to update it. But **record each change** with a date and the reason (in your lab notebook), and keep the old version.

---

### The Card
- **Question**: *the one question this sprint answers.*
- **Hypothesis**: *what you believe is true, and why (mechanism or prior evidence).*
- **Prediction**: *the concrete measurable outcome if the hypothesis is right: metric, direction, rough size.*
- **Falsifier**: *the result that would make you drop or revise the hypothesis.*
- **Null / baseline**: *what you compare against, and what "no effect" looks like.*
- **Stop criteria**: *when you stop or pivot (e.g. "no signal after N runs / X days", "baseline already solves it").*
- **Cheapest first test**: *the smallest experiment that could falsify it, ideally under a day.*

Stop criteria are set per sprint on the card. If you're unsure, start with: a 1-day time-box for the cheapest test, and a pivot if the falsifier holds on 3 seeds.

---

### @August 1, 2024 — [EXAMPLE CARD]
- **Question**: *Does adding Google Trends seasonality improve flu forecasting?*
- **Hypothesis**: *Our seasonal term is underfit because we don't have enough data; search volume is a proxy for the true seasonality.*
- **Prediction**: *≥2% lower MAPE on the held-out 2023 season.*
- **Falsifier**: *<0.5% improvement, or the gain disappears when the trends series is lagged by 2 weeks (that would point to leakage, not seasonality).*
- **Null / baseline**: *the current model with the Fourier seasonal term.*
- **Stop criteria**: *2 days; stop if the falsifier holds on 3 seeds.*
- **Cheapest first test**: *add the "flu" keyword only, at one region, with one seed.*
