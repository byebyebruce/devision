# pyright: reportOptionalSubscript=false
import os
import random
import sys
import zlib

sys.path.insert(0, os.path.dirname(__file__))
import vision_flan as vf  # noqa: E402


def item(task, prompt, answer, image="x.jpg"):
    return {"id": "t1", "task_name": task, "image": image,
            "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]}


def conv(task, prompt, answer):
    r, why = vf.convert(item(task, prompt, answer), "ext/vision_flan/images/x.jpg", "vision_flan:x.jpg",
                        random.Random(0))
    return r, why


def test_options_become_a_choice_with_the_lettered_answer():
    r, _ = conv("LSUN+Image_Classification", "<image>\nClassify.\nOptions: (a) tower (b) classroom (c) kitchen\n",
                "(c) kitchen")
    assert r["questions"]["q"]["type"] == "choice"
    assert set(r["questions"]["q"]["criteria"]) == {"tower", "classroom", "kitchen"}
    assert r["answer_key"] == "kitchen" and r["gold"]["q"]["probabilities"]["kitchen"] == 1.0
    assert r["questions"]["q"]["instructions"] == "What kind of place is this?"


def test_option_prefix_is_stripped_and_no_yes_options_become_noul():
    r, _ = conv("coco+image_classification_animal", "x | Options: (a) This image contains a cat "
                "(b) This image contains an elephant", "(b) This image contains an elephant")
    assert set(r["questions"]["q"]["criteria"]) == {"cat", "elephant"} and r["answer_key"] == "elephant"
    r, _ = conv("NUS-WIDE+Animal_classification", "Animal? | Options: (a) No (b) Yes", "(a) No")
    assert r["questions"]["q"]["type"] == "noul" and r["answer_key"] == "false"


def test_label_tasks_match_the_answer_sentence():
    r, _ = conv("Dark-Zurich+time_classification", "Options are: daytime, nighttime, twilight.",
                "The time of the day is twilight.")
    assert r["answer_key"] == "twilight" and len(r["questions"]["q"]["criteria"]) == 3
    r, _ = conv("SKETCH+living_organism_detection", "Living or not", "Non-Living")
    assert r["questions"]["q"]["type"] == "noul" and r["answer_key"] == "false"


def test_question_extracted_from_instruction_and_non_yes_no_dropped():
    assert vf.extract_question('This task tests counting. Here is the question "How many people are there?".') \
        == "How many people are there?"
    assert vf.extract_question("In this task, you will be asked about emotion. The question is Is the cat angry?") \
        == "Is the cat angry?"
    assert vf.extract_question("Question: Is this indoor? | Please answer by analyzing the scene.") == "Is this indoor?"
    r, _ = conv("VQA_object_presence", "Is there a cat in the picture? Decide if it appears.", "yes")
    assert r["questions"]["q"]["instructions"] == "Is there a cat in the picture?" and r["answer_key"] == "true"
    r, why = conv("VQA_scene_recognition", "Is this indoor or outdoor?", "indoor")
    assert r is None and why == "not a yes / no answer"


def test_count_words_become_nearby_number_options():
    r, _ = conv("VQA_counting", "The question is: How many dogs are there? What is the answer?", "three")
    assert r["answer_key"] == "3" and "3" in r["questions"]["q"]["criteria"]
    assert conv("VQA_counting", "How many people are there?", "fourteen")[0] is None


def test_caption_matching_reads_the_caption_and_the_match_option():
    r, _ = conv("ITM", 'Is the caption of image "A dog on a couch."? | Options: (a) not match (b) match', "(a) not match")
    assert r["questions"]["q"]["instructions"] == 'Does this caption describe the picture? "A dog on a couch."'
    assert r["answer_key"] == "false"
    r, _ = conv("ITM", 'Does "A cat on a bed." describe image? | Options: (a) the text is not a description of the '
                'image (b) the description matches the image', "(b) the description matches the image")
    assert r["answer_key"] == "true"


def test_coco_ids_from_file_names():
    assert vf.coco_id("VQA_counting_COCO_train2014_000000076081.jpg") == 76081
    assert vf.coco_id("coco+image_classification_animal_610_000000195449.jpg") == 195449
    assert vf.coco_id("GTSRB+image_classification_612_00010_00024.jpeg") is None


def test_excluded_tasks_get_a_reason():
    assert "test" in vf.excluded_reason("ImageNet-A+image_classification")
    assert "OCR" in vf.excluded_reason("DOCVQA+question_answer")
    assert vf.excluded_reason("CUB-200-2011+Bird_Classification") == vf.EXCLUDED_DEFAULT


def test_zip_members_are_read_out_of_a_coalesced_range():
    payloads = {"a.jpg": b"hello" * 50, "b.jpg": b"world" * 70}
    buf, index = b"", {}
    for name, data in payloads.items():
        comp = zlib.compressobj(6, zlib.DEFLATED, -15)
        raw = comp.compress(data) + comp.flush()
        extra = b"\x00" * 9
        header = (b"PK\x03\x04" + b"\x00" * 22 + len(name).to_bytes(2, "little") + len(extra).to_bytes(2, "little")
                  + name.encode() + extra)
        index[name] = [len(buf), len(raw), len(data), 8, zlib.crc32(data)]
        buf += header + raw
    ranges = vf.plan_ranges(list(index.items()), slack=64)
    assert len(ranges) == 1 and [n for n, _ in ranges[0][2]] == ["a.jpg", "b.jpg"]
    for name, e in index.items():
        assert vf.member_bytes(buf, 0, name, e) == payloads[name]
    bad = list(index["a.jpg"])
    bad[4] ^= 1
    assert vf.member_bytes(buf, 0, "a.jpg", bad) is None


def test_coco_distractors_present_in_the_photo_are_removed():
    r = conv("coco+image_classification_furniture", "x | Options: (a) This image contains a chair "
                "(b) This image contains a bed (c) This image contains a dining table", "(b) This image contains a bed")[0]
    assert r is not None
    out, _ = vf.prune_present(r, {"bed", "chair"})
    assert set(out["questions"]["q"]["criteria"]) == {"bed", "dining table"}
    assert out["gold"]["q"]["probabilities"] == {o: float(o == "bed") for o in out["questions"]["q"]["criteria"]}
    r = conv("coco+image_classification_furniture", "x | Options: (a) This image contains a chair "
                "(b) This image contains a bed", "(b) This image contains a bed")[0]
    assert r is not None
    assert vf.prune_present(r, {"chair"})[0] is None            # answer not annotated
    assert vf.prune_present(r, {"bed", "chair"})[0] is None     # one option left
