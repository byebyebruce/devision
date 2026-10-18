<!-- DRAFT (2026-10-17): the candidate is not locked; every number below is a placeholder filled from the locked
candidate's evaluation (release/v0.4/results.json, delivery_vs_v0.3.json). Nothing here is a measured result yet. -->
## training

TBD (filled when a candidate is locked; nothing below describes a trained model yet). The locked candidate's own
training recipe, steps, seed, trainable parts and input structure go here, read from its configs and train report.
If the candidate uses fixed local views (devision 0.3+: a full 256×256 picture giving 64 visual tokens, plus 12 local
views whose SigLIP patches are read by the question-conditioning module), say so and give the minimum package version;
otherwise describe its actual input. Do not reuse earlier draft recipes (the three-stage exp30-era plan, SigLIP's top
four layers fine-tuned, "input stays 256×256 with 64 visual tokens") unless the locked candidate is trained that way.

Temperatures are fitted on data/r13b/val.jsonl (3,892 questions) with devision-calibrate before the final freeze
(docs/experiments/2026-10-18-v04-local-release-prep.md); by question type and option-count bucket:

{{temperature_table}}

TBD: the delivery result against v0.3 (ScienceQA temperature + force dev questions, 68 : 114), the protection panels, and the link to [the delivery comparison](evaluation/delivery_vs_v0.3.json).

## limitations

- English, one image, yes/no and choice questions only. `score` is not implemented. At 256×256, small text and fine diagram details are hard to read.
- TBD: per-capability changes against v0.3 with intervals, including any protection panel that declined.
- The real-layout training pictures imitate two ScienceQA diagram forms; part of any gain on those forms may come from the similar layout. The development questions used to choose the recipe are not a blind test.
- Calibration reflects the calibration distribution, not every deployment domain. Validate thresholds on your own examples.
- Test sets have repeatedly informed experiment and release decisions. This is one training seed.
