"""GUI next-action questions: the gold is unique, distractors are actions on other elements of the same screen,
ambiguous / unlabeled steps are dropped, and held-out groups never reach train."""
import json

from v05_gui_action import (aitw_action, aitw_elements, aitw_group, aitw_question, cap_groups, crop_window,
                            gold_of, held_part, html_labels, m2w_question, parse_repr, sample, text_only)


def screen(*items):
    """AITW-style flat annotations from (label or ICON_X, x0, y0, x1, y1) tuples."""
    pos, texts, types = [], [], []
    for label, x0, y0, x1, y1 in items:
        pos += [y0, x0, y1 - y0, x1 - x0]
        icon = label.startswith("ICON_")
        texts.append("" if icon else label)
        types.append(label if icon else "TEXT")
    return aitw_elements(pos, texts, types)


SCREEN = screen(("Settings", 0.1, 0.1, 0.4, 0.15), ("Wi-Fi", 0.1, 0.2, 0.4, 0.25), ("Bluetooth", 0.1, 0.3, 0.4, 0.35),
                ("Display", 0.1, 0.4, 0.4, 0.45), ("ICON_MAGNIFYING_GLASS", 0.8, 0.05, 0.9, 0.1))


def tap_at(x, y):
    return {"results/action_type": 4, "touch_x": x, "touch_y": y, "lift_x": x, "lift_y": y}


def options_of(got):
    gold, options, n_screen = got
    return gold, options, n_screen


def test_tap_gold_is_the_touched_element_and_distractors_are_other_elements_of_the_screen():
    for k in range(20):
        got, why = aitw_question("q%d" % k, "turn on bluetooth", aitw_action(tap_at(0.2, 0.32)), SCREEN, [])
        assert got and not why
        gold, options, n_screen = got
        assert gold == 'tap "Bluetooth"' and options.count(gold) == 1
        assert 4 <= len(options) <= 6 and len({o.casefold() for o in options}) == len(options)
        on_screen = {'tap "Settings"', 'tap "Wi-Fi"', 'tap "Display"', "tap the magnifying glass icon"}
        assert n_screen >= 2 and len(set(options) & on_screen) == n_screen


def test_icon_gold_is_named_by_its_class():
    got, _ = aitw_question("q", "search", aitw_action(tap_at(0.85, 0.07)), SCREEN, [])
    assert got and got[0] == "tap the magnifying glass icon"


def test_unlabeled_ambiguous_duplicate_and_status_steps_are_dropped():
    assert aitw_question("q", "g", aitw_action(tap_at(0.7, 0.7)), SCREEN, [])[1] == "no_element"
    overlap = SCREEN + screen(("Bluetooth on", 0.15, 0.31, 0.3, 0.34))
    assert aitw_question("q", "g", aitw_action(tap_at(0.2, 0.32)), overlap, [])[1] == "ambiguous_element"
    dup = SCREEN + screen(("Bluetooth", 0.5, 0.8, 0.9, 0.85))
    assert aitw_question("q", "g", aitw_action(tap_at(0.2, 0.32)), dup, [])[1] == "duplicate_label"
    dots = screen(("...", 0.1, 0.1, 0.4, 0.15)) + SCREEN[1:]
    assert aitw_question("q", "g", aitw_action(tap_at(0.2, 0.12)), dots, [])[1] == "unreadable_label"
    assert aitw_question("q", "g", aitw_action({"results/action_type": 10}), SCREEN, [])[1] == "status_action"
    few = screen(("Settings", 0.1, 0.1, 0.4, 0.15), ("Wi-Fi", 0.1, 0.2, 0.4, 0.25))
    assert aitw_question("q", "g", aitw_action(tap_at(0.2, 0.12)), few, [])[1] == "few_screen_elements"


def test_scroll_type_and_navigation_golds():
    swipe_up = {"results/action_type": 4, "touch_x": 0.5, "touch_y": 0.8, "lift_x": 0.5, "lift_y": 0.3}
    assert aitw_action(swipe_up) == {"kind": "scroll", "phrase": "scroll down"}
    got, _ = aitw_question("q", "g", aitw_action(swipe_up), SCREEN, [])
    assert got and got[0] == "scroll down"
    typed = {"results/action_type": 3, "results/type_action": "pizza near me"}
    got, _ = aitw_question("q", "g", aitw_action(typed), SCREEN, ["pizza near me", "pizza"])
    assert got and got[0] == 'type "pizza near me"' and got[1].count('type "pizza near me"') == 1
    for k in range(30):   # back and home are never offered together
        got, _ = aitw_question("q%d" % k, "g", aitw_action({"results/action_type": 5}), SCREEN, [])
        assert got and got[0] == "go back" and "go to home screen" not in got[1]


def test_groups_and_split_keep_held_out_groups_out_of_train():
    from v05_gui_action import aitw_part
    assert aitw_group("aitw-shopping", "Go to ebay.com, search for a usb hub") == "site:ebay"
    assert aitw_group("aitw-shopping", "Look up the best rated coffee maker on Lowe's.") == "site:lowes"
    assert aitw_group("aitw-shopping", "Show me the best rated power washer on Home Depot.") == "site:homedepot"
    assert aitw_group("aitw-shopping", "Search for the new Nintendo switch on Walmart.") == "site:walmart"
    assert aitw_group("aitw-shopping", "Search for the best movies of 2019 ").startswith("goal:")
    assert aitw_part("aitw-shopping", "site:walmart") == "train" and aitw_part("aitw-shopping", "site:nike") == "test"
    assert aitw_part("aitw-shopping", "goal:search for the best movies of 2019") == "train"
    assert aitw_group("aitw-general", "What's the weather?") == aitw_group("aitw-general", "what's the  weather")
    rows = [{"group": "g%d" % (i % 50), "id": "r%d" % i} for i in range(500)]
    parts = {}
    for r in rows:
        parts.setdefault(held_part(r["group"], 0.2, 0.2), []).append(r)
    assert {r["group"] for r in parts["train"]}.isdisjoint({r["group"] for r in parts["dev"] + parts["test"]})
    kept, unused = cap_groups(parts["dev"], 25)
    assert len(kept) >= 25 and unused == len(parts["dev"]) - len(kept)
    assert len({r["group"] for r in kept}) * 10 == len(kept)   # whole groups only


HTML = """<html backend_node_id="1"><body>
<div backend_node_id="2"><a backend_node_id="3"><text backend_node_id="4">Flights</text></a>
<a backend_node_id="5"><span backend_node_id="6">Hotels</span></a>
<button backend_node_id="7">Search</button>
<input backend_node_id="8" placeholder="From"/>
<input backend_node_id="9" placeholder="To"/>
<a backend_node_id="10">Car rentals</a><a backend_node_id="11">Deals</a></div></body></html>"""


def cands(*rows):
    return [{"id": nid, "tag": tag, "box": box} for nid, tag, box in rows]


OTHERS = cands(("3", "a", (10, 10, 80, 30)), ("4", "text", (12, 12, 78, 28)), ("5", "a", (100, 10, 180, 30)),
               ("8", "input", (10, 100, 200, 130)), ("9", "input", (210, 100, 400, 130)),
               ("10", "a", (500, 10, 600, 30)), ("11", "a", (700, 10, 760, 30)),
               ("2", "div", (0, 0, 1280, 3000)))
WINDOW = (0, 0, 1280, 1000)


def test_html_labels_take_descendant_text_or_placeholder():
    labels = html_labels(HTML)
    assert labels["3"] == ("a", "Flights") and labels["5"] == ("a", "Hotels") and labels["8"] == ("input", "From")


def test_mind2web_click_and_type_questions():
    labels = html_labels(HTML)
    target = {"id": "7", "tag": "button", "box": (420, 100, 500, 130)}
    rep = parse_repr("[button]  Search -> CLICK")
    got, why = m2w_question("m1", rep, target, OTHERS, labels, WINDOW, ["New York"])
    assert got and not why
    gold, options, n_screen = got
    assert gold == 'click "Search"' and options.count(gold) == 1 and n_screen == 3
    # the page container (div) holds every label -> its text is too long, never an option
    assert not any("Flights Hotels" in o for o in options)
    target = {"id": "8", "tag": "input", "box": (10, 100, 200, 130)}
    rep = parse_repr("[textbox]  From -> TYPE: Boston")
    got, _ = m2w_question("m2", rep, target, OTHERS, labels, WINDOW, ["Boston"])
    assert got and got[0] == 'type "Boston" into "From"'
    assert 'type "Boston" into "To"' in got[1]   # same value, other field of the same screen


def test_mind2web_drops_duplicate_and_unlabeled_targets_and_out_of_window_elements():
    labels = html_labels(HTML)
    target = {"id": "3", "tag": "a", "box": (10, 10, 80, 30)}
    dup = OTHERS + cands(("99", "a", (10, 600, 80, 620)))
    labels_dup = dict(labels, **{"99": ("a", "Flights")})
    assert m2w_question("m", parse_repr("[a]  Flights -> CLICK"), target, dup, labels_dup, WINDOW, [])[1] \
        == "duplicate_label"
    assert m2w_question("m", parse_repr("[span]   -> CLICK"), target, OTHERS, labels, WINDOW, [])[1] \
        == "unreadable_label"
    # same label outside the crop window does not make the gold ambiguous and is not a distractor
    far = OTHERS + cands(("99", "a", (10, 1600, 80, 1620)))
    got, _ = m2w_question("m", parse_repr("[a]  Flights -> CLICK"), target, far, labels_dup, WINDOW, [])
    assert got and got[0] == 'click "Flights"'
    assert crop_window("m", (10, 1500, 80, 1520), 1280, 1400) is None
    w = crop_window("m", (10, 1500, 80, 1520), 1280, 4000)
    assert w and w[1] <= 1500 and w[3] >= 1520 and w[3] - w[1] == 1000


def test_text_only_report_counts_positions_and_heuristics():
    rows = [sample("q%d" % i, "s", "x.jpg", "s:%d" % i, "turn on bluetooth", [],
                   ['tap "Bluetooth"', 'tap "Wi-Fi"', "go back", "scroll down"][i % 4:] +
                   ['tap "Bluetooth"', 'tap "Wi-Fi"', "go back", "scroll down"][:i % 4], 'tap "Bluetooth"', {})
            for i in range(8)]
    rep = text_only(rows)
    assert rep["n"] == 8 and rep["goal_overlap"] == 1.0 and rep["chance"] == 0.25
    assert sum(rep["gold_position"].values()) == 8 and gold_of(rows[0]) == 'tap "Bluetooth"'
    json.dumps(rep)


def test_mind2web_skips_closed_menu_items_and_blank_or_crossing_boxes():
    import numpy as np
    from v05_gui_action import parse_html
    html = HTML.replace('<a backend_node_id="11">Deals</a>',
                        '<ul backend_node_id="20" role="menu"><a backend_node_id="21">Cruises</a></ul>'
                        '<a backend_node_id="11">Deals</a><a backend_node_id="30">Gift cards</a>'
                        '<a backend_node_id="31">Sign in</a>')
    labels, layered = parse_html(html)
    assert "21" in layered and "3" not in layered
    others = OTHERS + cands(("21", "a", (800, 10, 880, 30)), ("30", "a", (10, 300, 100, 330)),
                            ("31", "a", (60, 310, 140, 340)))   # 30 and 31 cross each other: a hidden layer
    pixels = np.zeros((1000, 1280), dtype=np.float32)
    pixels[::2] = 255.0                                        # text-like texture everywhere ...
    pixels[0:40, 690:770] = 0.0                                # ... except under "Deals"
    target = {"id": "7", "tag": "button", "box": (420, 100, 500, 130)}
    seen = set()
    for k in range(40):
        got, _ = m2w_question("m%d" % k, parse_repr("[button]  Search -> CLICK"), target, others, labels, WINDOW,
                              [], pixels, frozenset(layered))
        assert got
        seen |= set(got[1])
    assert not seen & {'click "Cruises"', 'click "Deals"', 'click "Gift cards"', 'click "Sign in"'}
    assert seen & {'click "Flights"', 'click "Hotels"', 'click "Car rentals"'}


def test_near_miss_taps_nav_bar_and_back_arrow_rules():
    # just outside the Wi-Fi box but inside it grown by GROW -> still Wi-Fi
    got, _ = aitw_question("q", "g", aitw_action(tap_at(0.42, 0.225)), SCREEN, [])
    assert got and got[0] == 'tap "Wi-Fi"'
    nav = SCREEN + screen(("ICON_NAV_BAR_CIRCLE", 0.45, 0.95, 0.55, 0.99))
    assert aitw_question("q", "g", aitw_action(tap_at(0.5, 0.97)), nav, [])[1] == "nav_bar_tap"
    back = nav + screen(("ICON_ARROW_BACKWARD", 0.0, 0.0, 0.08, 0.05))
    for k in range(20):
        got, _ = aitw_question("q%d" % k, "g", aitw_action({"results/action_type": 5}), back, [])
        assert got and "tap the arrow backward icon" not in got[1] and not any("nav bar" in o for o in got[1])
        got, _ = aitw_question("q%d" % k, "g", aitw_action(tap_at(0.04, 0.02)), back, [])
        assert got and got[0] == "tap the arrow backward icon" and "go back" not in got[1]


def test_scroll_cap_keeps_at_most_a_fifth_scroll_golds():
    from collections import Counter
    from v05_gui_action import cap_scroll
    opts = ["scroll down", 'tap "A"', 'tap "B"', 'tap "C"']
    rows = [sample("s%d" % i, "s", "x", "x", "g", [], opts, "scroll down", {}) for i in range(50)] + \
           [sample("t%d" % i, "s", "x", "x", "g", [], opts, 'tap "A"', {}) for i in range(40)]
    drops = Counter()
    out = cap_scroll(rows, drops)
    assert sum(gold_of(r) == "scroll down" for r in out) == 10 and len(out) == 50 and drops["scroll_cap"] == 40


def test_held_out_cap_spreads_over_groups_with_whole_episodes():
    rows = [{"group": "g%d" % (i % 5), "episode": "e%d" % (i % 25), "id": "r%d" % i} for i in range(500)]
    kept, unused = cap_groups(rows, 60, per_group=30)
    assert len({r["group"] for r in kept}) >= 2 and unused == 500 - len(kept)
    eps = {r["episode"] for r in kept}
    assert len(kept) == sum(1 for r in rows if r["episode"] in eps)   # episodes are never cut


def test_goal_whitespace_is_collapsed():
    from v05_gui_action import tidy
    r = sample("q", "s", "x", "x", 'Add Bio: "x"\n.', [], ['tap "A"', 'tap "B"'], 'tap "A"', {})
    assert "\n." not in r["questions"]["q"]["instructions"]
    r["questions"]["q"]["instructions"] = r["questions"]["q"]["instructions"].replace("Bio:", "Bio:\n\n")
    assert tidy(r)["questions"]["q"]["instructions"].startswith('Goal: Add Bio: "x" .\nWhich')
