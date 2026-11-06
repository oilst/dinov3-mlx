from profile_block import subprocess_lines


def test_subprocess_lines_gives_done_lines() -> None:
    stdout = 'log text\n{"side": 224, "batch_size": 1}\n{"side": 512, "batch_size": 1}\n'
    lines = subprocess_lines("ViT-S/16", "fp32", 0, stdout, "", {"need_bytes": 5})
    assert [line["side"] for line in lines] == [224, 512]
    assert all(line["status"] == "done" and line["need_bytes"] == 5 for line in lines)


def test_subprocess_lines_gives_an_error_line_for_a_failed_subprocess() -> None:
    stderr = "\n".join(f"line {i}" for i in range(50)) + "\nRuntimeError: MPS out of memory"
    lines = subprocess_lines("ViT-H+/16", "fp32", 1, "", stderr, {"need_bytes": 5})
    assert len(lines) == 1
    assert lines[0]["status"] == "error"
    assert lines[0]["model_size"] == "ViT-H+/16"
    assert "MPS out of memory" in lines[0]["reason"]
    assert "line 10" not in lines[0]["reason"]
    assert lines[0]["time"]
