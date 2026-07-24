from pathlib import Path

from nbs.analyze_neuroclaimbench_v21_feedback_crosswalk import _tree_sha256


def test_checkpoint_tree_hash_is_order_independent_and_content_sensitive(
    tmp_path: Path,
):
    first = tmp_path / "parent_0001.json"
    second = tmp_path / "parent_0002.json"
    first.write_text('{"claim_id":"a"}\n', encoding="utf-8")
    second.write_text('{"claim_id":"b"}\n', encoding="utf-8")

    original = _tree_sha256([second, first], tmp_path)

    assert original == _tree_sha256([first, second], tmp_path)
    second.write_text('{"claim_id":"changed"}\n', encoding="utf-8")
    assert original != _tree_sha256([first, second], tmp_path)
