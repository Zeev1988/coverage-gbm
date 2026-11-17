## Choosing a Research Topic

### Part 1: Is this a good topic?
Adapted from Ofir Press, [Tips for Finding Research Topics](https://ofir.io/Tips-for-Finding-Research-Topics/). Read the original; this is our paraphrase.

1. **Problem first, method second.** Start from a concrete problem the community already cares about. Build the dataset or setup around the problem. Bring in a method (RL, retrieval, a new architecture) only after results show it helps. Finding a good problem is most of the work.
2. **Iterate fast, keep solutions simple.** Results settle arguments that whiteboards can't, so get to an experiment early. The number of iterations matters more than how big each idea is. Always test the cheapest version first, using an oracle or upper bound: if a perfect component wouldn't help, the real one won't either.
3. **Write a paper people want to read.** It should do one of three things: teach something new about an existing system, improve on a benchmark, or introduce a benchmark. "Better" counts more than accuracy: training cost, inference cost, memory and latency all count. Aim at a topic a broad part of the field cares about.
4. **Aim at where the field will be about 9 months from now.** That's roughly how long a project takes. Check: are most of the papers you cite from the last 1–2 years? Predicting 9 months ahead is feasible; much further is guessing.
5. **Keep it simple.** The problem should be easy to state, the solution simple, and the code and paper easy for others to run and build on. Complex work gets accepted, but simple work is what lasts.

### Part 2: Has it been done?
1. Extract 3–5 **core claims** the work would make (the method, the problem, the mechanism, and why it beats the obvious baseline).
2. For each claim, search **arXiv, Semantic Scholar and OpenReview** with at least 3 different phrasings. Include the last 12 months.
3. Read the closest papers **in full text**, not just the abstract. **Verify every arXiv ID** by fetching it. Never cite a paper you haven't opened.
4. For each claim, record the **closest prior work**, the **remaining gap** in one sentence, and a verdict: **novel / incremental / done**.
5. Optional: ask Codex for an independent cross-review of the verdicts.

### Output
- **Novelty:** the verdict for each claim, with the closest prior work.
- **Feasibility:** compute, data, and the open checkpoints or code it depends on.
- **Impact:** which of the Part 1 paper types it is, and who would read it.
- **Likely reviewer score** (/10), with the main reason.
- **The narrowest version worth a hypothesis card.**
