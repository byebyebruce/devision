# v0.5 adoption

**Owner approved on 2026-10-30:** publish `runs/r24-v05/train` as model v0.5, after being informed that calibration is not fully preserved and the original formal ImageJevBench objectives have not been demonstrated. This is an explicit release trade-off, not a claim that all prewritten rules passed. The acceptance table below is unchanged.

| Rule (vs v0.4, paired by question, pictures resampled) | Result |
|---|---|
| 1. the five new uses (shopping screens, screen state, phone / web screens, first-person scenes, charts) improve with lower bound > 0 | met (+27.1 to +47.0) |
| 2. no other set significantly worse; POPE ≥ −1.0 point | met (0 of 24; POPE −0.5 [−1.1, +0.1]) |
| 3. ECE / NLL not worse than v0.4 | **not fully met**: NaturalBench NLL +0.029 [+0.021, +0.038]; ECE higher on 10 sets by 0.003–0.029 (largest RealWorldQA, MMBench, NaturalBench) |

The rule-3 failures are over-confidence on harder general benchmarks. Refitting the temperatures offline on the development sets only (dev_mix, or all development sets) brings NaturalBench to 0.705 / 0.698 (v0.4 0.693) and RealWorldQA below v0.4, but makes the size and counting sets worse; the released temperatures are the ones fitted at the end of training.

The POPE regression (−0.5) is within what the owner accepted on 2026-10-29. Test sets informed several decisions in this release cycle. This is one training seed.


The original ImageJevBench v0.3.0 admission, sub-1B leadership and five official domain objectives are not established by this release. The project-specific five-use evaluation and the public Decision Index are different protocols. No official ImageJevBench rank is claimed.
