"""Round 10: the four diagram question types are named by their question text; everything else stays out."""
from v10_build import diagram_type


def test_the_four_types_and_nothing_else():
    assert diagram_type("Will these magnets attract or repel each other?") == "repel"
    assert diagram_type("Think about the magnetic force between the magnets in each pair. Which of the following "
                        "statements is true?") == "force"
    assert diagram_type("Compare the average kinetic energies of the particles in each sample. Which sample has the "
                        "higher temperature?") == "temp"
    assert diagram_type("Which solution has a higher concentration of  purple particles?") == "conc"
    assert diagram_type("Complete the text to describe the diagram. Solute particles moved in both directions") is None
    assert diagram_type("Which of these states is farthest north?") is None


def test_held_files_include_round_10s_monitoring_set(tmp_path):
    from v5_build import held_files
    for d in ("v2", "v3", "v4", "v6", "lv_bench", "v10"):
        (tmp_path / d).mkdir()
    (tmp_path / "v10" / "dev_sqa_diagram.jsonl").write_text("")
    (tmp_path / "v10" / "train_x.jsonl").write_text("")
    files = [p.replace(str(tmp_path), "") for p in held_files(str(tmp_path))]
    assert "/v10/dev_sqa_diagram.jsonl" in files and "/v10/train_x.jsonl" not in files
