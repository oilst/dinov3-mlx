"""Rewrite the history of a new clone for the public push (D7 B, Section 3.3 of
specs/07-release/plan.html), then check it (AC15, AC16).

Run it in a new clone, never in the working folder: it changes every commit id.

    uv run python scripts/rewrite_history.py --patterns <file> --email <address> [--repo <clone>]

The pattern file lives outside the repo, because it holds the private strings. One rule per line:

    <literal>\\t<replacement>      a literal string and its replacement (\\n and \\t are escapes)
    re:<regex>\\t<replacement>     a regular expression and its replacement (with group refs)
    check:<regex>                 a regular expression that must have 0 hits after the rewrite
    # ...                         a comment

The run: git filter-branch with an env filter (author and committer address), a tree filter (the
rules on each text file; in *.html and *.md files each token of 7 to 12 hex digits that names an
old commit becomes the first 7 digits of its new id) and a message filter (the rules). Each doc
cites only older commits, so their new ids are in the map folder of filter-branch when the tree
filter runs. Then the checks:

- AC16: the new history has the commit count and the subjects of the old history; the HEAD trees
  differ only in the rules and the cited ids; each cited id resolves to the commit with the subject
  of the old cited commit.
- AC15: after the old refs are gone (refs/original, reflog, gc), no object of the repo holds a hit
  of a rule or of a home folder path, and each author and committer address is the new address.

The script contains no private string.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# The prefix of a home folder on macOS: never in the public history (AC15).
HOME_PREFIX = "~/"
HEX_TOKEN = re.compile(r"\b[0-9a-f]{7,12}\b")
CITING_SUFFIXES = (".html", ".md")
NEW_ID_LENGTH = 7


class ReleaseCheckError(RuntimeError):
    """The rewritten history fails a check of AC15 or AC16."""


@dataclass(frozen=True)
class Rule:
    """One line of the pattern file.

    Attributes:
        search: The compiled search pattern.
        replacement: The replacement, or None for a check term.
        literal: True if the replacement is a plain string (no group refs).
    """

    search: re.Pattern
    replacement: str | None
    literal: bool = True

    def apply(self, text: str) -> str:
        """Return text with each hit replaced."""
        if self.replacement is None:
            return text
        if self.literal:
            return self.search.sub(lambda _: self.replacement, text)
        return self.search.sub(self.replacement, text)


def unescape(text: str) -> str:
    """Turn the escapes \\n and \\t of a pattern line into a newline and a tab."""
    return text.replace("\\n", "\n").replace("\\t", "\t")


def read_rules(path: Path) -> list[Rule]:
    """Read the pattern file.

    Raises:
        ValueError: For a rule line without a tab.
    """
    rules = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith("check:"):
            rules.append(Rule(re.compile(line.removeprefix("check:")), None))
            continue
        if "\t" not in line:
            raise ValueError(f"{path}:{number}: a rule needs a tab between search and replacement")
        search, replacement = line.split("\t", 1)
        if search.startswith("re:"):
            rules.append(Rule(re.compile(search.removeprefix("re:")), replacement, literal=False))
        else:
            rules.append(Rule(re.compile(re.escape(unescape(search))), unescape(replacement)))
    return rules


def apply_rules(rules: list[Rule], text: str) -> str:
    """Apply each rule in the order of the file."""
    for rule in rules:
        text = rule.apply(text)
    return text


def cited_commit(token: str, old_ids: list[str]) -> str | None:
    """Return the old commit that a hex token names, or None if it names none or several."""
    hits = [full for full in old_ids if full.startswith(token)]
    return hits[0] if len(hits) == 1 else None


def map_ids(text: str, old_ids: list[str], new_id: Callable[[str], str | None]) -> str:
    """Replace each token that names an old commit with the first digits of its new id.

    Args:
        text: The text of an *.html or *.md file.
        old_ids: The full ids of the old commits.
        new_id: The new full id of an old full id, or None if it is not known yet.
    """

    def replace(match: re.Match) -> str:
        old = cited_commit(match.group(), old_ids)
        new = new_id(old) if old else None
        return new[:NEW_ID_LENGTH] if new else match.group()

    return HEX_TOKEN.sub(replace, text)


def transform(
    name: str, data: bytes, rules: list[Rule], old_ids: list[str], new_id: Callable
) -> bytes:
    """Return the new content of one file: the rules on text, the id map on *.html and *.md.
    Binary content stays."""
    if b"\0" in data[:8192]:
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    new = apply_rules(rules, text)
    if name.endswith(CITING_SUFFIXES):
        new = map_ids(new, old_ids, new_id)
    return new.encode("utf-8") if new != text else data


def tree_filter(rules_path: Path, map_dir: Path, ids_path: Path) -> None:
    """The tree filter of filter-branch: change the files of the current folder in place."""
    rules = read_rules(rules_path)
    old_ids = json.loads(ids_path.read_text())

    def new_id(old: str) -> str | None:
        path = map_dir / old
        return path.read_text().split()[0] if path.exists() else None

    for folder, _, names in os.walk("."):
        for name in names:
            path = Path(folder) / name
            if path.is_symlink():
                continue
            data = path.read_bytes()
            new = transform(path.as_posix(), data, rules, old_ids, new_id)
            if new != data:
                path.write_bytes(new)


def git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    """Run git in repo and return its output."""
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True, env=env
    ).stdout


def git_bytes(repo: Path, *args: str) -> bytes:
    """Run git in repo and return its output as bytes."""
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True).stdout


def check_trees(repo: Path, old_head: str, rules: list[Rule], id_map: dict[str, str]) -> dict:
    """AC16: the HEAD trees differ only in the rules and the cited ids; each cited id resolves."""
    old_ids = list(id_map)
    old_names = git(repo, "ls-tree", "-r", "--name-only", old_head).splitlines()
    new_names = git(repo, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    if old_names != new_names:
        raise ReleaseCheckError("the HEAD trees hold different files")
    differ, cited, unresolved = [], 0, []
    subject = lambda commit: git(repo, "log", "-1", "--format=%s", commit).strip()
    for name in old_names:
        old = git_bytes(repo, "show", f"{old_head}:{name}")
        new = git_bytes(repo, "show", f"HEAD:{name}")
        if transform(name, old, rules, old_ids, id_map.get) != new:
            differ.append(name)
        if not name.endswith(CITING_SUFFIXES) or b"\0" in old[:8192]:
            continue
        for token in HEX_TOKEN.findall(old.decode("utf-8", errors="ignore")):
            commit = cited_commit(token, old_ids)
            if commit is None:
                continue
            cited += 1
            short = id_map[commit][:NEW_ID_LENGTH]
            resolved = subprocess.run(
                ["git", "rev-parse", "--verify", "-q", f"{short}^{{commit}}"],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            expected = apply_rules(rules, subject(commit))
            if resolved != id_map[commit] or subject(resolved) != expected:
                unresolved.append(f"{name}: {token}")
    if differ:
        raise ReleaseCheckError(f"the HEAD trees differ in more than the rules: {differ[:10]}")
    if unresolved:
        raise ReleaseCheckError(f"{len(unresolved)} cited ids do not resolve: {unresolved[:10]}")
    return {"cited_ids": cited, "unresolved": 0}


def check_objects(repo: Path, rules: list[Rule], email: str, count: int) -> dict:
    """AC15: no object holds a hit; each address is the new address; the commit count stays."""
    listing = git(repo, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)")
    ids = listing.split()
    raw = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=repo,
        input="\n".join(ids).encode() + b"\n",
        capture_output=True,
        check=True,
    ).stdout
    searches = [rule.search for rule in rules] + [re.compile(re.escape(HOME_PREFIX))]
    hits: dict[str, int] = {}
    position = 0
    for _ in ids:
        header_end = raw.index(b"\n", position)
        size = int(raw[position:header_end].split()[2])
        text = raw[header_end + 1 : header_end + 1 + size].decode("utf-8", errors="ignore")
        position = header_end + 1 + size + 1
        for search in searches:
            if search.search(text):
                hits[search.pattern] = hits.get(search.pattern, 0) + 1
    if hits:
        found = ", ".join(f"{pattern!r}: {n} objects" for pattern, n in hits.items())
        raise ReleaseCheckError(f"the history still holds private strings: {found}")
    addresses = git(repo, "log", "--all", "--format=%ae%n%ce").split()
    if set(addresses) != {email} or len(addresses) != 2 * count:
        raise ReleaseCheckError(f"addresses other than {email}: {sorted(set(addresses) - {email})}")
    commits = len(git(repo, "rev-list", "--all").split())
    if commits != count:
        raise ReleaseCheckError(f"the new history has {commits} commits, the old one {count}")
    return {"objects": len(ids), "addresses": len(addresses)}


def rewrite(repo: Path, patterns: Path, email: str) -> dict:
    """Rewrite the history of repo and check it.

    Args:
        repo: A new clone. Its branches are rewritten in place.
        patterns: The pattern file.
        email: The author and committer address of every commit.

    Returns:
        The report: commit count, cited ids, objects and addresses checked.

    Raises:
        ReleaseCheckError: If a check of AC15 or AC16 fails.
    """
    rules = read_rules(patterns)
    branch = git(repo, "symbolic-ref", "HEAD").strip()
    old_ids = git(repo, "rev-list", "--reverse", branch).split()
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        ids_path = work / "old-ids.json"
        ids_path.write_text(json.dumps(old_ids))
        script = f"{shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).resolve()))}"
        rewrite_dir = work / "rewrite"
        quoted = shlex.quote(str(patterns.resolve()))
        tree = (
            f"{script} tree-filter --patterns {quoted} "
            f"--map-dir {shlex.quote(str(rewrite_dir / 'map'))} --ids {shlex.quote(str(ids_path))}"
        )
        address = shlex.quote(email)
        env_filter = f"export GIT_AUTHOR_EMAIL={address} GIT_COMMITTER_EMAIL={address}"
        environ = os.environ | {"FILTER_BRANCH_SQUELCH_WARNING": "1"}
        git(
            repo,
            "filter-branch",
            "-f",
            "-d",
            str(rewrite_dir),
            "--env-filter",
            env_filter,
            "--tree-filter",
            tree,
            "--msg-filter",
            f"{script} msg-filter --patterns {quoted}",
            "--",
            "--branches",
            "--tags",
            env=environ,
        )
    new_ids = git(repo, "rev-list", "--reverse", branch).split()
    if len(new_ids) != len(old_ids):
        raise ReleaseCheckError(f"{len(new_ids)} new commits against {len(old_ids)} old commits")
    id_map = dict(zip(old_ids, new_ids, strict=True))
    report = {"commits": len(new_ids)}
    report |= check_trees(repo, f"refs/original/{branch}", rules, id_map)
    for ref in git(repo, "for-each-ref", "--format=%(refname)", "refs/original").split():
        git(repo, "update-ref", "-d", ref)
    git(repo, "reflog", "expire", "--expire=now", "--all")
    git(repo, "gc", "-q", "--prune=now")
    report |= check_objects(repo, rules, email, len(old_ids))
    return report


def main(argv: list[str] | None = None) -> None:
    """Parse the arguments: the rewrite, or one of the two filters of filter-branch."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("tree-filter", "msg-filter"):
        parser = argparse.ArgumentParser(prog=f"rewrite_history.py {argv[0]}")
        parser.add_argument("--patterns", type=Path, required=True)
        parser.add_argument("--map-dir", type=Path)
        parser.add_argument("--ids", type=Path)
        args = parser.parse_args(argv[1:])
        if argv[0] == "tree-filter":
            tree_filter(args.patterns, args.map_dir, args.ids)
        else:
            sys.stdout.write(apply_rules(read_rules(args.patterns), sys.stdin.read()))
        return
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--patterns", type=Path, required=True, help="the pattern file")
    parser.add_argument("--email", required=True, help="the author and committer address")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="the new clone")
    parser.add_argument("--report", type=Path, help="write the report as JSON to this file")
    args = parser.parse_args(argv)
    report = rewrite(args.repo, args.patterns, args.email)
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
