"""Test T15 of specs/07-release/plan.html: scripts/rewrite_history.py on a small fake repo."""

import os
import subprocess
from pathlib import Path

import pytest
import rewrite_history

OLD_EMAIL = "someone@old.example"
NEW_EMAIL = "someone@new.example"
NOT_A_COMMIT = "deadbeef42"  # a hex token that names no commit
HOME = "/" + "Users/"  # the home folder prefix, in two parts: the public history must not hold it


def git(repo: Path, *args: str) -> str:
    """Run git in repo and return its output."""
    env = os.environ | {
        "GIT_AUTHOR_NAME": "Someone",
        "GIT_AUTHOR_EMAIL": OLD_EMAIL,
        "GIT_COMMITTER_NAME": "Someone",
        "GIT_COMMITTER_EMAIL": OLD_EMAIL,
    }
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True, env=env
    ).stdout.strip()


def commit(repo: Path, files: dict[str, str], message: str) -> str:
    """Write files, commit them and return the full id of the new commit."""
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def fake_repo(tmp_path: Path) -> tuple[Path, list[str]]:
    """A repo with 4 commits. Commit 1 holds a private word and a home folder path; the docs of
    commits 2 and 4 cite older commits."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    c1 = commit(
        repo,
        {"a.txt": "made for SecretCo\n", "log.jsonl": f'{{"reason": "{HOME}someone/x failed"}}\n'},
        "Commit one",
    )
    c2 = commit(repo, {"docs/notes.md": f"Fixed in {c1[:7]}. Hash {NOT_A_COMMIT}.\n"}, "Commit two")
    c3 = commit(repo, {"a.txt": "made for SecretCo, again\n"}, "Commit three")
    c4 = commit(repo, {"docs/page.html": f"<p>See {c2[:10]} and {c1[:7]}.</p>\n"}, "Commit four")
    return repo, [c1, c2, c3, c4]


@pytest.fixture
def patterns(tmp_path: Path) -> Path:
    path = tmp_path / "patterns.txt"
    path.write_text(
        "# a literal, a regular expression and a check term\n"
        "SecretCo\tthe client\n"
        f"re:{HOME}[a-z]+/\t~/\n"
        "check:(?i)secretco\n"
    )
    return path


def test_rewrite_history(fake_repo: tuple[Path, list[str]], patterns: Path) -> None:
    """T15: the pattern leaves every commit, the address changes, cited ids map to the new ids,
    other hex tokens stay, and the checks pass. The run works on a clone, as in W9."""
    source, old = fake_repo
    repo = source.parent / "clone"
    git(source.parent, "clone", "-q", "--no-local", str(source), str(repo))
    report = rewrite_history.rewrite(repo, patterns, NEW_EMAIL)
    new = git(repo, "rev-list", "--reverse", "HEAD").split()
    assert len(new) == 4 and set(new).isdisjoint(old)
    assert report["commits"] == 4 and report["cited_ids"] == 3 and report["unresolved"] == 0
    log = git(repo, "log", "--all", "--format=%ae %ce")
    assert set(log.split()) == {NEW_EMAIL}
    everything = git(repo, "log", "--all", "-p", "--format=%B")
    assert "SecretCo" not in everything and HOME not in everything
    assert git(repo, "show", "HEAD:a.txt") == "made for the client, again"
    notes = git(repo, "show", f"{new[1]}:docs/notes.md")
    assert notes == f"Fixed in {new[0][:7]}. Hash {NOT_A_COMMIT}."
    page = git(repo, "show", "HEAD:docs/page.html")
    assert page == f"<p>See {new[1][:7]} and {new[0][:7]}.</p>"
    assert git(repo, "for-each-ref", "refs/original", "refs/remotes") == ""
    assert git(repo, "remote") == ""


def test_rewrite_history_check_finds_a_planted_miss(
    fake_repo: tuple[Path, list[str]], patterns: Path, tmp_path: Path
) -> None:
    """T15: the check fails when a string of the pattern file stays in the history."""
    repo, _ = fake_repo
    partial = tmp_path / "partial.txt"
    partial.write_text(f"re:{HOME}[a-z]+/\t~/\ncheck:(?i)secretco\n")  # misses SecretCo
    with pytest.raises(rewrite_history.ReleaseCheckError, match="(?i)secretco"):
        rewrite_history.rewrite(repo, partial, NEW_EMAIL)


def test_rewrite_history_check_finds_a_home_folder(
    fake_repo: tuple[Path, list[str]], tmp_path: Path
) -> None:
    """T15: the check fails when a home folder path stays, also without a pattern for it."""
    repo, _ = fake_repo
    partial = tmp_path / "partial.txt"
    partial.write_text("SecretCo\tthe client\n")
    with pytest.raises(rewrite_history.ReleaseCheckError, match=HOME):
        rewrite_history.rewrite(repo, partial, NEW_EMAIL)
