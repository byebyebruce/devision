"""v0.4 ChartQA subset: the eligibility filter keeps a direct reading and rejects max / difference / label questions,
distractors come from the same column and never equal the gold, and the mismatched control is a derangement that
keeps the original image_id."""
from v04_chartqa import cell_check, distractors, make_row, mismatched, text_reason

TWO_COL = [["Characteristic", "Number of stores"], ["2016", "39"], ["2017", "41"], ["2018", "45"], ["2019", "45.0*"],
           ["2020", "47"], ["2021", "52"]]
MULTI = [["Characteristic", "Male", "Female"], ["2018", "12.5", "13.1"], ["2019", "12.9", "13.5"],
         ["2020", "13.2", "13.9"]]


def eligible(query, label, table):
    if text_reason(query, label):
        return None
    cell, _ = cell_check(query, label, table)
    return cell


def test_direct_read_is_accepted():
    cell = eligible("How many stores did the company operate in 2020?", "47", TWO_COL)
    assert cell is not None and (cell["row"], cell["column"], cell["raw"]) == ("2020", "Number of stores", "47")
    cell = eligible("What was the female population of Madagascar in 2019?", "13.5", MULTI)
    assert cell is not None and (cell["row"], cell["column"]) == ("2019", "Female")


def test_max_difference_label_and_ambiguous_questions_are_rejected():
    assert text_reason("Which year had the highest number of stores?", "2021")
    assert text_reason("What was the highest number of stores?", "52")
    assert text_reason("What is the difference between stores in 2020 and 2016?", "8")
    assert text_reason("In what year did the company operate 47 stores?", "2020")
    assert text_reason("Is the number of stores in 2020 above 40?", "Yes")
    assert text_reason("What was the total number of stores in 2019 and 2020?", "92")
    # numeric but not a direct cell: wrong row named, gold in no cell, two equal cells, series not named
    assert eligible("How many stores did the company operate in 2016?", "47", TWO_COL) is None
    assert eligible("How many stores did the company operate in 2020?", "48", TWO_COL) is None
    assert eligible("How many stores were there in 2018?", "45", TWO_COL) is None   # 45 and 45.0* -> two cells
    assert eligible("What was the population of Madagascar in 2019?", "13.5", MULTI) is None


def test_distractors_come_from_the_same_column_and_never_equal_the_gold():
    for qid in ("a", "b", "c", "d"):
        cell, _ = cell_check("How many stores did the company operate in 2020?", "47", TWO_COL)
        assert cell is not None
        ds = distractors(TWO_COL, cell, "47", qid)
        assert len(ds) == 3
        assert all(c["from"] == "same_column" and c["column"] == "Number of stores" for _, c in ds)
        options = [o for o, _ in ds]
        assert "47" not in options and len(set(options)) == 3
    cell, _ = cell_check("What was the female population of Madagascar in 2019?", "13.5", MULTI)
    assert cell is not None
    ds = distractors(MULTI, cell, "13.5", "x")
    assert [c["from"] for _, c in ds] == ["same_column", "same_column", "other_column"]
    assert {c["column"] for _, c in ds[:2]} == {"Female"}
    row = make_row("multi_col_1.png", 0, "What was the female population of Madagascar in 2019?", "13.5",
                   ["13.5"] + [o for o, _ in ds], cell)
    probs = row["gold"]["q"]["probabilities"]
    assert len(probs) == 4 and [k for k, v in probs.items() if v == 1.0] == ["13.5"]


def test_mismatched_is_a_derangement_and_keeps_the_original_image_id():
    rows = [{"id": "v04-chartqa:c%d.png:0" % i, "image_id": "chartqa:c%d.png" % i,
             "image": "v04_chartqa/images/c%d.png" % i} for i in range(7)]
    mism, move = mismatched(rows, 0)
    assert sorted(move) == sorted(move.values()) == sorted(r["image_id"] for r in rows)
    assert all(k != v for k, v in move.items())
    for r, m in zip(rows, mism):
        assert m["image_id"] == m["mismatched_from"] == r["image_id"]
        assert m["shown_image_id"] == move[r["image_id"]] != r["image_id"]
        assert m["image"] == "v04_chartqa/images/%s" % m["shown_image_id"].split(":", 1)[1]
    assert mismatched(rows, 0) == (mism, move)
