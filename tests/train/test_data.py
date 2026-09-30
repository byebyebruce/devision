"""Converter seam: raw dataset records -> Jev-format training samples."""
# pyright: reportOptionalSubscript=false
from devision.train.data import convert_gqa, convert_pope, convert_vqav2, select

GQA_VERIFY = {
    "imageId": "2354786", "question": "Is there a dog in the picture?", "answer": "yes",
    "types": {"structural": "verify", "semantic": "obj", "detailed": "existThat"},
    "semantic": [{"operation": "select", "argument": "dog (-)", "dependencies": []},
                 {"operation": "exist", "argument": "?", "dependencies": [0]}],
}
GQA_CHOOSE = {
    "imageId": "2400001", "question": "Is the man standing or sitting?", "answer": "sitting",
    "types": {"structural": "choose", "semantic": "attr", "detailed": "chooseAttr"},
    "semantic": [{"operation": "select", "argument": "man (1123)", "dependencies": []},
                 {"operation": "choose pose", "argument": "standing|sitting", "dependencies": [0]}],
}
GQA_CHOOSE_REL = {
    "imageId": "2400002", "question": "Is the cup to the left or to the right of the plate?",
    "answer": "to the right of",
    "types": {"structural": "choose", "semantic": "rel", "detailed": "chooseRel"},
    "semantic": [{"operation": "select", "argument": "plate (77)", "dependencies": []},
                 {"operation": "choose rel", "argument": "cup,to the left of|to the right of,s (78)",
                  "dependencies": [0]}],
}
GQA_OPEN = {
    "imageId": "2400003", "question": "What color is the car?", "answer": "red",
    "types": {"structural": "query", "semantic": "attr", "detailed": "colorQuery"},
    "semantic": [{"operation": "select", "argument": "car (5)", "dependencies": []},
                 {"operation": "query", "argument": "color", "dependencies": [0]}],
}


def only_question(sample):
    (q,) = sample["questions"].values()
    (gold,) = sample["gold"].values()
    return q, gold


def test_gqa_yes_no_question_becomes_noul():
    sample = convert_gqa("201307251", GQA_VERIFY)

    q, gold = only_question(sample)
    assert q == {"type": "noul", "instructions": "Is there a dog in the picture?"}
    assert gold["probabilities"] == {"false": 0.0, "true": 1.0}
    assert sample["image_id"] == "vg:2354786"
    assert sample["image"] == "gqa/images/2354786.jpg"
    assert sample["source"] == "gqa"


def test_gqa_choose_question_becomes_two_option_choice():
    q, gold = only_question(convert_gqa("1", GQA_CHOOSE))

    assert q["type"] == "choice"
    assert q["instructions"] == "Is the man standing or sitting?"
    assert set(q["criteria"]) == {"standing", "sitting"}
    assert gold["probabilities"] == {"standing": 0.0, "sitting": 1.0}


def test_gqa_relation_choice_reads_options_from_the_program():
    q, gold = only_question(convert_gqa("2", GQA_CHOOSE_REL))

    assert set(q["criteria"]) == {"to the left of", "to the right of"}
    assert gold["probabilities"]["to the right of"] == 1.0


def test_gqa_open_question_is_skipped():
    assert convert_gqa("3", GQA_OPEN) is None


def test_gqa_choose_whose_answer_is_not_an_option_is_skipped():
    broken = dict(GQA_CHOOSE, answer="lying")
    assert convert_gqa("4", broken) is None


def vqa(answer_type, answers, mc=None, image_id=9, qid=90):
    question = {"image_id": image_id, "question": "Is the dog sitting?", "question_id": qid}
    ann = {"image_id": image_id, "question_id": qid, "answer_type": answer_type,
           "multiple_choice_answer": mc or answers[0],
           "answers": [{"answer": a, "answer_confidence": "yes", "answer_id": i + 1}
                       for i, a in enumerate(answers)]}
    return question, ann


def test_vqav2_yes_no_question_becomes_noul_with_annotator_agreement_as_target():
    q, gold = only_question(convert_vqav2(*vqa("yes/no", ["yes"] * 7 + ["no"] * 3, mc="yes"),
                                          split="train2014"))

    assert q == {"type": "noul", "instructions": "Is the dog sitting?"}
    assert gold["probabilities"] == {"false": 0.3, "true": 0.7}


def test_vqav2_sample_points_at_the_coco_image():
    sample = convert_vqav2(*vqa("yes/no", ["no"] * 10, image_id=42), split="val2014")

    assert sample["image_id"] == "coco:42"
    assert sample["image"] == "coco/val2014/COCO_val2014_000000000042.jpg"


def test_vqav2_non_yes_no_question_is_skipped():
    assert convert_vqav2(*vqa("number", ["2"] * 10), split="train2014") is None
    assert convert_vqav2(*vqa("other", ["red"] * 10), split="train2014") is None


def noul_samples(n_yes, n_no, first_image=0):
    out = []
    for i in range(n_yes + n_no):
        answers = ["yes"] * 10 if i < n_yes else ["no"] * 10
        out.append(convert_vqav2(*vqa("yes/no", answers, image_id=first_image + i, qid=i),
                                 split="train2014"))
    return out


def test_selection_never_contains_an_evaluation_image():
    samples = noul_samples(20, 20)
    held_out = {"coco:3", "coco:25", "coco:39"}

    picked = select(samples, limit=40, exclude_image_ids=held_out, seed=0)

    assert picked
    assert not {s["image_id"] for s in picked} & held_out


def test_selection_balances_yes_and_no():
    picked = select(noul_samples(30, 6), limit=10, exclude_image_ids=set(), seed=0)

    yes = sum(s["gold"]["q"]["probabilities"]["true"] > 0.5 for s in picked)
    assert len(picked) == 10
    assert yes == 5


def test_selection_is_reproducible_for_a_seed():
    samples = noul_samples(30, 30)

    ids = lambda seed: [s["id"] for s in select(samples, limit=10, exclude_image_ids=set(), seed=seed)]  # noqa: E731
    assert ids(1) == ids(1)
    assert ids(1) != ids(2)


def test_selection_keeps_choice_questions():
    choices = [convert_gqa(str(i), dict(GQA_CHOOSE, imageId=str(i))) for i in range(4)]

    picked = select(noul_samples(3, 3) + choices, limit=100, exclude_image_ids=set(), seed=0)

    assert sum(s["questions"]["q"]["type"] == "choice" for s in picked) == 4


def test_pope_question_becomes_noul_on_its_coco_val_image():
    rec = {"question_id": "7", "question": "Is there a snowboard in the image?", "answer": "no",
           "image_source": "COCO_val2014_000000310196", "category": "adversarial"}

    sample = convert_pope(rec)

    q, gold = only_question(sample)
    assert q == {"type": "noul", "instructions": "Is there a snowboard in the image?"}
    assert gold["probabilities"] == {"false": 1.0, "true": 0.0}
    assert sample["image_id"] == "coco:310196"
    assert sample["image"] == "coco/val2014/COCO_val2014_000000310196.jpg"
    assert sample["id"] == "pope:adversarial:7"


def test_gqa_relation_answer_given_as_a_single_word_matches_its_option():
    # Real GQA: the program says "to the right of", the answer is just "right".
    rec = dict(GQA_CHOOSE_REL, answer="right")

    q, gold = only_question(convert_gqa("5", rec))

    assert gold["probabilities"] == {"to the left of": 0.0, "to the right of": 1.0}
