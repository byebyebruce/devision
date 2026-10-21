from n1_mix import smooth


def sample(p):
    return {"id": "x", "gold": {"q": {"probabilities": p}}}


def test_smoothing_moves_only_one_hot_gold():
    hard = smooth(sample({"false": 0.0, "true": 1.0}), 0.1)["gold"]["q"]["probabilities"]
    assert abs(hard["false"] - 0.05) < 1e-9 and abs(hard["true"] - 0.95) < 1e-9
    four = smooth(sample({"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0}), 0.1)["gold"]["q"]["probabilities"]
    assert abs(four["a"] - 0.925) < 1e-9 and abs(four["b"] - 0.025) < 1e-9
    soft = {"false": 0.3, "true": 0.7}
    assert smooth(sample(soft), 0.1)["gold"]["q"]["probabilities"] == soft
    assert smooth(sample({"false": 0.0, "true": 1.0}), 0.0)["gold"]["q"]["probabilities"] == {"false": 0.0, "true": 1.0}
