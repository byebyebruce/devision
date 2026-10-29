"""KIE questions: value distractors are other amounts / answers of the same document, the gold is unique, unparseable
fields are not asked, and yes / no is balanced per field."""
from collections import Counter

from v05_board_kie import balance_present, cord_questions, funsd_questions, parse_amount

RECEIPT = {"menu": [{"nm": "ICED TEA", "price": "12,000"}, {"nm": "NASI GORENG", "price": "35,000"},
                    {"nm": "KERUPUK", "price": "5,000"}],
           "sub_total": {"subtotal_price": "52,000", "tax_price": "5,200", "discount_price": "abc"},
           "total": {"total_price": "57,200", "cashprice": "60,000", "changeprice": "2,800"}}


def test_parse_amount():
    assert parse_amount("Rp. 12.000") == 12000 and parse_amount("1,234.50") == 1234.5 and parse_amount("x") is None


def test_cord_value_options_come_from_the_same_receipt():
    printed = {12000, 35000, 5000, 52000, 5200, 57200, 60000, 2800}
    qs = cord_questions("train-0", RECEIPT, Counter(), Counter())
    values = [q for q in qs if q[1] == "value"]
    assert {q[2] for q in values} >= {"total_price", "item_price", "changeprice", "tax_price", "subtotal_price"}
    for qid, kind, fld, q, opts, gold in values:
        assert len(opts) == 4 and opts.count(gold) == 1
        assert {int(o.replace(",", "")) for o in opts} <= printed
    total = next(q for q in values if q[2] == "total_price")
    assert total[5] == "57,200"


def test_cord_presence_skips_unparseable_fields():
    pres = {q[2]: q[5] for q in cord_questions("train-0", RECEIPT, Counter(), Counter()) if q[1] == "present"}
    assert pres["tax_price"] == "Yes" and pres["service_price"] == "No"
    assert "discount_price" not in pres                     # printed but unparseable


def test_balance_present_equalises_yes_and_no_per_field():
    qs = [("a%d" % i, "tax", "Yes") for i in range(10)] + [("b%d" % i, "tax", "No") for i in range(4)] + \
         [("c%d" % i, "cash", "No") for i in range(3)]
    keep, dropped = balance_present(qs)
    assert len([q for q in keep if q.startswith("a")]) == 4 and len([q for q in keep if q.startswith("b")]) == 4
    assert not [q for q in keep if q.startswith("c")] and dropped["present_unbalanced:tax"] == 6


def ent(i, label, text, box, links=()):
    return {"id": i, "label": label, "text": text, "box": box, "linking": [list(l) for l in links]}


def test_funsd_value_and_presence():
    ents = [ent(0, "question", "DATE:", [10, 10, 60, 20], [(0, 1)]), ent(1, "answer", "5/12/1990", [70, 10, 140, 20]),
            ent(2, "answer", "6/1/1991", [70, 100, 140, 110]), ent(3, "answer", "7/7/1992", [70, 200, 140, 210]),
            ent(4, "answer", "8/8/1993", [70, 300, 140, 310]), ent(5, "question", "FAX NO.", [10, 500, 60, 510]),
            ent(6, "answer", "ACME", [400, 700, 460, 710])]
    qs = funsd_questions("doc", ents, Counter())
    value = [q for q in qs if q[1] == "value"]
    assert len(value) == 1 and value[0][5] == "5/12/1990" and value[0][4].count("5/12/1990") == 1
    assert set(value[0][4]) == {"5/12/1990", "6/1/1991", "7/7/1992", "8/8/1993"}
    pres = {q[3]: q[5] for q in qs if q[1] == "present"}
    assert pres == {'On this form, is the field "DATE" filled in?': "Yes",
                    'On this form, is the field "FAX NO." filled in?': "No"}
