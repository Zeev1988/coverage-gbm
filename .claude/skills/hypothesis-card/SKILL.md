---
name: hypothesis-card
description: Use when starting a research sprint or project, choosing between research ideas, or when a hypothesis is vague and needs a concrete prediction, falsifier, and stop criteria; also when checking results against a pre-stated hypothesis.
---

Read `docs/guides/HYPOTHESIS-CARD.TEMPLATE.md` and fill in every field of the card.

- The **Prediction** and the **Falsifier** must be measurable: a metric, a direction, and a threshold. If the user can't give one, say so instead of inventing it.
- If the **Stop criteria** field is empty, ask the user for it. If they have no preference, use the fallback from the guide. Never leave it blank.
- The **Cheapest first test** should take under a day. If none exists, the hypothesis is probably too broad. Suggest a narrower one.
- When results come in, check them against the card and say plainly whether the prediction held or the falsifier fired. Log any change to the card with the `lab-notebook` skill.
- Once the card is filled in, draft `docs/research/PREREGISTRATION.md` from `docs/guides/PREREGISTRATION.TEMPLATE.md`. Committing it before the first real run freezes it.
- Next steps: `abstract-v0`, then `main-result-figure`.
