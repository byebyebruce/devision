<!-- DRAFT (2026-10-19, V04-K-MATERIALS-PREP-01): the candidate is frozen (runs/v04-k/cand, runs/v04-k/eval/IDENTITY.json
c14863e2...) and its synthetic confirmation is done; the full evaluation is running. Every "TBD" below is filled from
the sealed full evaluation (release/v0.4/results.json, delivery_vs_v0.3.json) -- nothing marked TBD is a measured result
yet, and this draft is not a release. -->
## training

v0.4 is a **reading combination**, not a single network: (1) for the two supported diagram forms, an automatic
region-reading step looks at the picture and reads a few printed numbers into a small text table, (2) a fixed,
hand-written rule computes one derived quantity from them, and (3) the decision model answers those questions from the
request's own text plus that table (and K) **without the picture**; every other question goes to the vision decision
model with the picture as before. The three parts are bound and verified together
(`reading/` in the checkpoint, with its lineage); `devision.load(...)` runs them by default (`reading="auto"`).

- **Decision model.** Continues v0.3 (r15) through the exp30–protect-k chain of LoRA / head training runs (alignment
  replay, comparison and fact training, the K training of exp72 and the protection training protect-k, one epoch,
  LoRA r16, seed 1); LoRA is merged. Input: a 256×256 letterboxed picture giving 64 visual tokens, plus 12 fixed local
  views (overlap 0.25) read by the question-conditioning module. The exact stage configs are in the round directories
  and in [the training configuration](evaluation/training.yaml) (TBD: copied at build).
- **Reading step.** Six fixed question templates in two ScienceQA diagram forms: particle mass / speed of samples A and B
  (temperature questions) and the distance between the magnets of pairs 1 and 2 (magnetic-force questions). A selector
  picks the region, a character head reads the digits and a corrector widens the speed boxes (the components of
  EXP71). Other questions are not read and go to the decision model directly.
- **K rule (deterministic).** For temperature questions, when all four values are readable whole numbers, the rule
  writes K = mass × speed² for each sample into the table (otherwise no K line). This is ordinary arithmetic done by
  code; the model did **not** learn to square numbers.

Temperatures were fitted with devision-calibrate on data/r13b/val.jsonl (3,892 questions) by question type and
option-count bucket. Questions the reading step actually read have their own temperature per diagram form, fitted on
1,200 synthetic calibration pictures (out-of-fold check before adoption):

{{temperature_table}}

On the ScienceQA temperature (41) and magnet (120) development questions -- used during development, so not a blind
test -- accuracy rose by 29.4 points against v0.3 [21.1, 37.7] (temperature 41/41). TBD: the release decision and
the links to the delivery comparison and full evaluation tables. **Status: not release-eligible under the current
rules (the dev16 retention requirement is not met); this draft is not a release.**

## limitations

- English, one image, yes/no and choice questions only. `score` is not implemented. At 256×256, small text and fine
  diagram details are hard to read. Requires devision 0.3 or newer (reading combination, local views, read-question
  temperatures); earlier packages cannot load this model.
- **Narrow reading.** Only the two ScienceQA diagram forms above are read, through fixed templates. Development
  numbers on those forms (LIVE-K: temperature comparison 93.0% on 300 questions; reading questions 96.7% as
  multiple choice, raw values right 93.7%) come from pictures in the same layouts used during development; they are not
  an independent test.
- **New drawings of the same forms read poorly.** On a separately rendered synthetic confirmation (120 new pictures,
  480 questions) the combination beat v0.3 by +13.3 points on the comparison question (95% interval +4.2 to +23.3)
  and gained +12.5 points from seeing the right picture (+2.5 to +23.3). But the comparison accuracy itself was only
  46.7%; "equal" cases were almost always wrong (2/15 temperature, 2/20 magnets). The reading questions' 53.3%
  (temperature) and 80.0% (magnets) are multiple-choice accuracies; the raw numbers the reading step wrote matched the
  true values for only 1 of 240 temperature values (47 unreadable) and 57 of 120 magnet distances (14 unreadable) --
  for example it placed speeds in the mass fields. Treat the reading as layout-specific.
- **Option order matters.** Reversing the options changed the chosen answer on 106 of 300 temperature questions
  (38 wrong→right, 41 right→wrong, 27 wrong→wrong) and 21 of 180 magnet questions; similar overall accuracy hides
  these per-question changes.
- Against v0.3 (paired by question, picture-clustered 95% intervals, sealed full evaluation): the seven protection
  panels did not trigger the release rule's cost line, while several individual sets fell slightly -- the held-out GQA binding test -2.2
  [-3.9, -0.5], position -2.0, size -1.6, VQA choice -1.5; What'sUp rose +4.9 [+2.3, +7.4]. The synthetic two-panel
  reading / comparison task used as a development monitor (dev16, families never trained on) is **not reached** on
  this model's branch: 36.8% reading, 41.5% comparison, below its 80% rule. It was shown learnable on a separate
  experimental branch (EXP26) that this model does not descend from; the reading step does not cover it.
- Chart questions (80 ChartQA-style items): 31.3% with the right picture and 36.3% with a wrong one; a max-value
  baseline gets 61.3%. There is no evidence that the model reads charts.
- Two held-out sets each share one picture with sets watched during development (test_count_fresh: coco:221272;
  test_bind_gqa: coco:418606); they are not fully independent. Several other sets, including ScienceQA, informed
  development and are reported as monitoring, not as blind tests. The project's training pictures were not re-scanned
  for leakage for this release.
- Calibration reflects the calibration distribution, not every deployment domain; the read-question temperatures were
  fitted on synthetic pictures. Validate thresholds on your own examples.
- Test sets have repeatedly informed experiment and release decisions. This is one training seed.
