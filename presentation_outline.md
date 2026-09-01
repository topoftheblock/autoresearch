# Autoresearch Loops: A Causal Study of the Instruction File
## 15-Minute Defense Presentation — Slide Outline & Speaker Script

**Christian Block** · Research project with Prof. Ramesh
14 slides · ~60 seconds per slide · ~14 min speaking + 1 min buffer

---

### A note on two points in the brief

**Temperature was 0.7, not 0.0.** The thesis argues explicitly that greedy decoding would have been the wrong choice: run-to-run variance *is* the error term every coefficient is tested against, so a deterministic loop drives it to zero and takes every standard error with it. Slide 9 presents 0.7 as a deliberate decision, because a committee will ask about it and the answer is a strength.

**"ReAct / Ralph" is the audience's vocabulary, not the paper's.** The thesis describes a propose → execute → interpret → decide cycle. Slide 5 names that shape and notes the resemblance without claiming the paper uses the term.

---

## SLIDE 1 — Nobody knows why these loops work

**Visual:** Full-bleed screenshot of the `karpathy/autoresearch` GitHub page, star and fork counts visible. No text overlay.

**On screen:**
- March 2026: autoresearch ships
- ~13,000 forks in six months
- An agent runs its own experiments
- Nobody can say which instruction worked

**Speaker notes (~55s):**

In March 2026, Andrej Karpathy released a repository called autoresearch. The idea is simple and slightly unsettling. You give a language model a goal, some code it is allowed to edit, and a script that runs a trial and hands back one number. Then you walk away. Overnight the model proposes a change, runs it, reads the result, decides what to try next, and repeats until the budget is gone. In six months it has been forked over thirteen thousand times. People are running it on real systems. And here is the part that motivated this project: when one of these loops produces a good result, nobody can tell you which part of the instructions was responsible. That is the gap I set out to close.

---

## SLIDE 2 — The architecture is three files

**Visual:** Three boxes side by side, each labelled with a filename and an owner badge. Arrows from `program.md` into the loop. Colour the owner badges differently: NOBODY / AGENT / HUMAN.

**On screen:**
- `prepare.py` — data and score, nobody edits
- `train.py` — the model, agent edits
- `program.md` — instructions, human edits
- No scheduler. No driver program.

**Speaker notes (~55s):**

The architecture is a contract between three files, and what matters is who owns each one. `prepare.py` fixes the data and defines the score. Nobody touches it, so every experiment is scored identically. `train.py` holds the model, and the agent is the only thing that edits it. `program.md` holds the instructions, and the human is the only one who edits that. Now notice what is missing. There is no scheduler. There is no driver program. The nine-step experiment cycle exists only as English prose inside `program.md`, and the model itself is the execution layer. So `program.md` is not configuration handed to a controller. It *is* the controller. That is the observation the whole project rests on.

---

## SLIDE 3 — It already works in production. That is the problem.

**Visual:** Screenshot of Shopify/liquid PR #2056 header showing the title and the "Open" status badge. Pull one number out large: **53%**.

**On screen:**
- Shopify Liquid, PR #2056
- 53% faster parse and render
- ~120 experiments, 93 commits
- Author: "probably overfitted"
- Never merged

**Speaker notes (~60s):**

This is not a toy concern. Here is an autoresearch run against Shopify's Liquid template engine, which is production infrastructure. It cut parse and render time by fifty-three per cent, over roughly a hundred and twenty experiments across ninety-three commits. And then it was never merged. The author's own assessment was that the gain was probably overfitted. So we have a system that produced a large, real, measurable improvement, and the people closest to it could not decide whether to trust it. Benchmarks have the same shape. The AI Scientist, Coscientist, MLAgentBench — all of them report whether the pipeline succeeded end to end. None of them tell you which part of the instruction file did the work. Success is measured; attribution is not.

---

## SLIDE 4 — Two questions

**Visual:** Clean slide, two numbered questions in large type. Nothing else.

**On screen:**
- **RQ1** — Which components cause which behaviour, and how large?
- **RQ2** — Do the components act independently?

**Speaker notes (~50s):**

The obstacle is confounding. An instruction file settles several things at once: what counts as success, how widely the agent may search, when it must stop, how it must report, where to spend effort. If you compare two differently worded files, every sentence differs, so nothing is identified. And the prompt-sensitivity literature makes this sharper — Sclar and colleagues showed that formatting changes which preserve meaning can move accuracy by seventy-six points. Wording clearly matters, and reading the wording will not tell you how. So I ask two questions. First: which components of the instruction file causally determine which aspects of the loop's behaviour, and how large is each effect? Second: do those components act independently, or does the effect of one depend on another?

---

## SLIDE 5 — The loop under study

**Visual:** The flow diagram from the thesis (Figure 1). Shade the two language-model boxes; leave the executor box unshaded. Add a legend: shaded = model, unshaded = deterministic.

**On screen:**
- Propose → Execute → Interpret → Decide
- Model chooses. Harness measures.
- Search space: hyperparameters, not code
- Every number is traceable

**Speaker notes (~60s):**

This is the loop. The model alternates between two moves, each a single JSON object. A propose carries a hypothesis, a model family, and hyperparameter values. An interpret carries a reading of the last measurement and a decision to continue or stop. If you have seen ReAct or the Ralph pattern, this is that shape — reason, act, observe, repeat. One narrowing matters. The search space is a hyperparameter space, not arbitrary code, so that the instruction file stays the only free variable. And critically, the model never computes a number. It picks what gets computed, and reads the answer back. The shaded boxes are model calls; the unshaded one is a deterministic harness running scikit-learn. That means every value in a transcript is traceable: either the executor produced it, and it is a fact about the task, or the model produced it, and it is a fact about the loop's behaviour.

---

## SLIDE 6 — Stop reading the prompt. Start configuring it.

**Visual:** An annotated `program.md` on the left with five paragraphs highlighted in five colours; on the right, the vector `c = (1,0,1,0,1)` with each digit colour-matched to its slot.

**On screen:**
- One template, five variable slots
- Everything else byte-identical
- Each slot written at two levels
- A file becomes a vector

**Speaker notes (~55s):**

Here is the core move of the project. Instead of treating the instruction file as prose to be interpreted, I treat it as a configuration vector. Every file comes out of one template. The fixed part — the research question, the executor interface, the response format — is byte-identical across every file I generate. The variable part is exactly five paragraphs, each written at one of two levels. Nothing else in the file moves. So a complete instruction file is described by five binary digits. That is what makes the problem tractable: once a prompt is a vector, the question "what does this file do" becomes a question you can answer with an experiment instead of an argument.

---

## SLIDE 7 — The five components

**Visual:** Five-row table, one row per component, columns Level 0 / Level 1. Keep the wording short; the full text is in the appendix.

**On screen:**
- **M** Evaluation criterion — vague vs. precise
- **B** Search breadth — one family vs. both
- **S** Stopping rule — fixed 3 vs. adaptive 8
- **O** Reporting style — terse vs. full sentences
- **E** Emphasis — explore vs. exploit first

**Speaker notes (~60s):**

These are the five components. Evaluation criterion: either success is left undefined, or it is pinned to five-fold cross-validated accuracy. Search breadth: commit to one model family, or compare both. Stopping rule: a fixed budget of exactly three experiments, or an adaptive budget of up to eight that halts on diminishing returns. Reporting style: terse, or full sentences stating hypothesis and interpretation. And emphasis: explore several configurations first, or exploit a promising one immediately. These are not invented for the study. A real `program.md` hardcodes a baseline, says how to handle failures, fixes a reporting discipline, and imposes a parsimony rule. These five come from those categories. One control worth noting: a language model wrote all ten paragraphs from fixed meta-prompts, and they were then frozen and reused word for word — so my own stylistic habits are not a free parameter tangled up in the treatment.

---

## SLIDE 8 — Run the whole space

**Visual:** A 2×2×2×2×2 lattice or a compact 32-row grid of ±1 patterns, with the equation N = 2⁵ × D × R = 1280 underneath.

**On screen:**
- All 2⁵ = 32 configurations
- × 2 datasets × 20 replicates
- **N = 1,280 real runs**
- Constructed, not observed

**Speaker notes (~55s):**

I run the complete space. All thirty-two configurations, crossed with two datasets, twenty replicates each — twelve hundred and eighty real runs against a real executor. Two things follow from running all of it. First, because I construct the configurations rather than collecting them, confounding is eliminated by the design itself, with no statistical adjustment. In observational prompt data, whoever writes a precise success criterion probably also writes a careful stopping rule — those two would be correlated. Construction breaks that. Second, with all thirty-two files in hand, every pairwise interaction is estimated separately from every main effect. A fractional design would have to assume the components act independently, because the runs that distinguish a main effect from an interaction are exactly the ones it skips. Thirty-two cells is the price of not assuming, and at this scale I can pay it.

---

## SLIDE 9 — What I held fixed, and one thing I deliberately did not

**Visual:** Two columns. Left, "Held constant" with a lock icon. Right, a single highlighted box: "Temperature 0.7 — on purpose."

**On screen:**
- Pinned model snapshot, one seed list
- One stratified 70/30 split
- Same executor, same whitelist
- **Temperature 0.7, not 0.0**
- Variance is the error term

**Speaker notes (~60s):**

Controls. The agent model is pinned to a dated snapshot rather than a moving alias. The split, the seed, the executor, the whitelist, the fixed part of the template — all constant. Runs are separate conversations, so nothing carries over. Now the one that gets questioned, so let me address it directly. I did not run at temperature zero. It is fixed at zero point seven for every single run. This is deliberate. Run-to-run variance is the error term that every coefficient in this study is tested against. A deterministic loop would drive that variance to zero and take every standard error with it — I would have point estimates and no way to say whether any of them were distinguishable from noise. So I hold temperature constant and above zero, which makes it a controlled constant rather than an eliminated one, and I pass an explicit sampling seed so any individual run can still be regenerated exactly.

---

## SLIDE 10 — Four metrics, straight off the transcript

**Visual:** Four icons in a row — a target, a percentage, an upward step, a stopwatch — each with its metric name. No judge model, no framework: say it visually with a crossed-out robot icon.

**On screen:**
- Gain over default — did it improve anything?
- Wasted trial ratio — malformed + duplicate
- Improvement rate — trials beating the best so far
- Cost to best — calls to reach the peak
- No judge model. No scoring library.

**Speaker notes (~50s):**

Four dependent variables, all computed mechanically from the structured transcript. Gain over default: the score of the configuration the run finally recommends, minus a default-hyperparameter baseline computed before any run. Wasted trial ratio: proposals that were malformed or rejected by the whitelist, plus exact duplicates of earlier proposals, over all attempts — both of those burn a turn without buying a measurement. Improvement rate: the share of executed trials that beat the best score seen earlier in the same run. And cost to best: how many executor calls it took to reach the peak. No evaluation framework, no scoring library, no judge model is involved anywhere. Every metric is a difference in accuracy, a share of proposals, or a count of calls.

---

## SLIDE 11 — The model, and why the dataset is in it

**Visual:** The regression equation rendered large and clean, with the γⱼDⱼ term visually highlighted in a second colour.

**On screen:**
- Y = β₀ + Σ βᵢXᵢ + Σ γⱼDⱼ + ε
- ±1 coding, orthogonal by construction
- γⱼ absorbs task difficulty
- HC3 errors, Benjamini–Hochberg

**Speaker notes (~55s):**

Each component is coded plus or minus one, and I regress each metric on those five indicators — plus a dataset fixed effect, the gamma term highlighted here. That term is not decoration. Breast Cancer and Wine differ in how much headroom they leave and how they respond to the whitelist. Without the dataset in the model, that variation sits in the residual, inflates every standard error, and makes real effects harder to see. Because the factorial is crossed with the task suite, each component indicator is orthogonal to the dataset indicator by construction, so the coefficients I report are within-task effects. I fit on individual runs rather than cell means, use heteroskedasticity-consistent standard errors — the stopping rule pins the experiment count at one level and frees it at the other, so the variance genuinely is not constant — and apply Benjamini–Hochberg for multiplicity.

---

## SLIDE 12 — What actually moved

**Visual:** The coefficient forest plot from the thesis (Figure 2), four panels. Animate or highlight the Wasted-trial-ratio panel first, since that is where the story is.

**On screen:**
- Reporting style dominates: −0.075
- Wasted trials: 10.6% → 3.1%
- Stopping rule: the only outcome effect
- Cost to best: nothing survives

**Speaker notes (~60s):**

Here is what actually moved. The wasted trial ratio is the metric the instruction file really governs — four of the five components shift it, and the model explains sixteen per cent of its variance. Reporting style produces the largest effect anywhere in the design: minus zero point zero seven five. In plain terms, requiring the model to write full sentences cuts wasted trials from about ten and a half per cent to three per cent. That is a two-thirds reduction, from a change in wording alone. The stopping rule is the only component that touches either outcome metric — worth about one tenth of a point on the gain, and it lifts the improvement rate from thirteen and a half to seventeen per cent. And the cost to best answers to nothing: no component survives adjustment. The largest candidate, exploit-first, lands at p equals zero point zero five seven. So close, and I am not going to claim it.

---

## SLIDE 13 — The task matters more than the instructions

**Visual:** A simple paired bar chart: largest component effect vs. dataset effect, for Improvement rate and Cost to best. The dataset bar dwarfs the other in both.

**On screen:**
- Wine vs. Breast Cancer: −0.064 improvement rate
- Cost to best: −0.20
- Both at p < 10⁻⁵
- Larger than any component effect
- Adaptive rule obeyed in 20% of runs

**Speaker notes (~60s):**

Two findings worth dwelling on. First: on two of the four metrics, the dataset effect is roughly twice the largest instruction component. Moving from Breast Cancer to Wine shifts the improvement rate by six and a half points and the cost to best by two tenths of a call, both at p below ten to the minus five. Which task you pick matters more than how you word the prompt. That is exactly the variation the fixed effect removes from the residual. Second, and this constrains everything else: I measured whether the model actually obeyed each instruction. The fixed budget is followed in eighty-nine per cent of runs. The adaptive rule, in twenty. So the stopping-rule effects I just showed are the effect of *offering* an adaptive budget, not of a loop that reliably stops on diminishing returns. Every estimate here is intention-to-treat, attenuated toward zero.

---

## SLIDE 14 — What we proved

**Visual:** Three lines of text, generous whitespace. No chart. Let it land.

**On screen:**
- Instructions govern **process**, not outcome
- Verbose reporting: −2/3 wasted trials
- Components act independently (3 of 40)
- The method transfers to any sliced prompt

**Speaker notes (~60s):**

So, what did we prove. The instruction file governs how efficiently the loop conducts its search, not the quality of what it finds. Requiring the model to explain itself removes two thirds of the wasted proposals — a large and reproducible effect. But no wording tested here made the loop find a better model, and I would argue that is a ceiling effect: untuned defaults already reach ninety-six per cent. On RQ2, three of forty interactions survive adjustment, all on one metric and all involving reporting style, so the components can be treated as independent — and because the factorial is complete, that is something I tested rather than assumed. Finally, the method is not tied to the five components I happened to vary. Any instruction file that can be cut into slots can be studied this way. Thank you. I am happy to take questions.

---

## Appendix — anticipated questions

| Question | Answer |
|---|---|
| Why not temperature 0? | It would zero the error term. Slide 9. |
| R² is only 0.16 — is that weak? | For a 5-factor design over noisy agent behaviour, yes and expected. The effects are small relative to run-to-run variance; that is why N = 1280. |
| Only one agent model? | Stated limitation. Findings are conditional on `gpt-4o-mini-2024-07-18`; this measures how instructions act on one policy, not a property of instructions in general. |
| One wording per level? | Stated limitation. Component effect is confounded with its particular phrasing. Sampling several wordings per level is the top follow-up. |
| Why does M have no compliance rate? | The executor computes the score whatever the instruction says, so M leaves no behavioural trace. O gets a manipulation check instead: 205 vs. 386 characters. |
| Is 40% improving on baseline bad? | It is a ceiling effect, not a failure. a₀ = 0.960 on both tasks. |
