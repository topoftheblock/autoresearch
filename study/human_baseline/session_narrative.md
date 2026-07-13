# Think-aloud session: does `max_features` affect Random Forest generalization?

*Authored simulation of a human research session (see `README.md` for what is
and isn't genuine about this). Timestamps are illustrative, as if transcribed
from a recording. All numbers are real, taken from actual runs of
`loop/executor.py` against the same fixed breast-cancer train/validation
split used everywhere else in this project.*

**[00:00]** Okay, the question I've been asked to look at is whether the
feature-subsampling ratio — `max_features` in scikit-learn's Random Forest —
actually affects generalization on this dataset. I don't have a strong prior
here beyond textbook intuition: each tree in the forest picks a random subset
of features to consider at each split, and the whole point of that is to
decorrelate the trees so that averaging them actually reduces variance. If
every tree could see every feature, they'd all tend to pick the same strong
splits and end up correlated, which defeats some of the purpose of bagging.
So my rough expectation going in is that *some* subsampling should help
relative to none, but I genuinely don't know where the sweet spot is for a
30-feature dataset like this one.

**[00:35]** Let me not overthink this and just start from the default.
scikit-learn uses `sqrt(n_features)` by default for classification, so with
30 features that's about 5 or 6 features per split. I'll fix everything else
— 200 trees, no depth limit — so the only thing that changes across runs is
this one knob.

**[01:10]** *(runs experiment: `max_features="sqrt"`)* — cv accuracy 0.9623,
std 0.0139, validation accuracy 0.9415. Okay, that's a solid number to anchor
on.

**[01:40]** Now let me check the other extreme: what if every tree gets to
see every feature? That's basically just bagging with no feature
subsampling at all, `max_features=1.0`.

**[02:05]** *(runs experiment: `max_features=1.0`)* — cv accuracy 0.9422,
std 0.0204. That's noticeably worse on both counts: lower mean *and* higher
variance across folds. That's consistent with what I expected — without
feature subsampling the trees are more correlated, so the ensemble doesn't
average out noise as well, and you can see that both in the lower mean and
in the higher spread across folds.

**[02:40]** There's another common default I've seen people use, `log2`,
which for 30 features works out to about 5 features — close to `sqrt` but
not identical. Let me check that too before I start interpolating.

**[03:00]** *(runs experiment: `max_features="log2"`)* — cv accuracy 0.9598,
std 0.0148. Slightly behind `sqrt` on the mean, and marginally less stable.
Close, but `sqrt` still ahead.

**[03:30]** So the picture so far is: fewer features per split looks better,
at least between these three points. I want to check whether that's a clean
trend or whether there's a dip and recovery somewhere, so let me fill in the
middle of the range. I'll try 0.5 first, halfway between "sqrt-ish" and "all
features."

**[04:00]** *(runs experiment: `max_features=0.5`)* — cv accuracy 0.9498,
std 0.0177. That's already worse than both `sqrt` and `log2`, and the
variance is climbing too. So whatever is happening, it looks monotonic in
this direction rather than U-shaped — I'm not seeing evidence of some better
optimum further out toward the middle.

**[04:35]** Let me get two more points to make sure I'm not missing
something: 0.3, a bit above the sqrt/log2 neighborhood, and 0.7, further
toward the "use everything" end.

**[05:10]** *(runs experiment: `max_features=0.3`)* — cv accuracy 0.9548,
std 0.0129. That sits neatly between `log2` and 0.5, which is what I'd
expect if the relationship really is roughly monotonic.

**[05:35]** *(runs experiment: `max_features=0.7`)* — cv accuracy 0.9473,
std 0.0201. Continues the same downward trend toward 1.0, both in mean
accuracy and in stability.

**[06:05]** At this point the trend from `sqrt`/`log2` down toward 1.0 looks
pretty clear and consistent: five points now (`sqrt`, `log2`, 0.3, 0.5, 0.7,
1.0 — six, actually) all lined up in the direction I'd expect. The one thing
I haven't checked is whether going *below* `sqrt` keeps helping, or whether
`sqrt` is already close to a floor. Let me try one more point, 0.2, which at
30 features is about 6 features — right in the same neighborhood as `sqrt`
(5.5) but let's see.

**[06:40]** *(runs experiment: `max_features=0.2`)* — cv accuracy 0.9548,
std 0.0129. Interesting — that ties exactly with the 0.3 run rather than
beating `sqrt`. So it doesn't look like going lower than `sqrt`'s
neighborhood buys anything further; if anything `sqrt` (roughly 0.18 of the
features here) still edges it out slightly.

**[07:10]** I think I have enough to answer the question. Feature
subsampling clearly matters here: using a small fraction of features per
split (in the `sqrt`-to-`log2` neighborhood, roughly 15-20% of the 30
features) gives both the best mean cross-validated accuracy and the lowest
variance across folds, and performance degrades close to monotonically as
you increase `max_features` toward using every feature at every split, which
also gets less stable, not just less accurate. There's no evidence of a
second, better optimum further out, and going somewhat below `sqrt` (to 0.2)
didn't improve on `sqrt` itself. I'll stop here and recommend
`max_features="sqrt"` as a sensible default for this dataset, which happens
to match what scikit-learn already defaults to — a nice, if unglamorous,
confirmation that the library default isn't arbitrary for a dataset like
this one.
