<!-- DRAFT (2026-10-17): the candidate is not locked; every number below is a placeholder filled from the locked
candidate's evaluation (release/v0.4/results.json, delivery_vs_v0.3.json). Nothing here is a measured result yet. -->
## training

This release continues v0.3 (`r15-vision`) in three stages:

1. **Alignment with real-caption replay** (2,500 steps): masked caption completion on 20,000 synthetic number-reading pictures mixed 1:1 with 20,000 COCO train2014 photos and their captions. Every image used by an evaluation, calibration or monitoring set is excluded by image id (COCO and Visual Genome ids matched) and path.
2. **Real-layout alignment** (1,500 steps): captions listing the drawn values by label, on 12,000 newly drawn pictures in the layout of two ScienceQA diagram forms (gas samples in jars; pairs of magnets), mixed 1:1 with real COCO captions. The pictures are synthetic; no ScienceQA picture is used, and value combinations of the diagnostic questions are never drawn.
3. **Decision training** (TBD steps, seed 1): v0.3's decision mix as replay (12,000 questions), reading and ScienceQA-form questions on 1,200 real-layout families (each a picture and a twin that changes only the values), and 2,401 synthetic panel questions. SigLIP2's top four layers are fine-tuned at 5e-6 as in v0.3. LoRA is merged into the released weights. Input stays 256×256 with 64 visual tokens.

Temperatures were fitted on the designated calibration set, by question type and option-count bucket:

{{temperature_table}}

TBD: the delivery result against v0.3 (ScienceQA temperature + force dev questions, 68 : 114), the protection panels, and the link to [the delivery comparison](evaluation/delivery_vs_v0.3.json).

## limitations

- English, one image, yes/no and choice questions only. `score` is not implemented. At 256×256, small text and fine diagram details are hard to read.
- TBD: per-capability changes against v0.3 with intervals, including any protection panel that declined.
- The real-layout training pictures imitate two ScienceQA diagram forms; part of any gain on those forms may come from the similar layout. The development questions used to choose the recipe are not a blind test.
- Calibration reflects the calibration distribution, not every deployment domain. Validate thresholds on your own examples.
- Test sets have repeatedly informed experiment and release decisions. This is one training seed.
