# A rewritten history with valid commit ids, and a new working folder

The first 88 commits had private strings in old versions of some files, local home folder paths in 4 commits, and a company author address in every commit. The docs cite about 130 commit ids. The user wants no private string and no company name in the public repo (Q4 and Q5 of `specs/07-release/explore.html`).

Decision (D7 B, the user, 2026-11-09; P2 of `plan.html`, the user, 2026-11-10):

1. **Rewrite in a new clone.** `scripts/rewrite_history.py` runs `git filter-branch` in a new clone (`git clone --no-local`). The tree filter replaces the private strings in each text file of each commit. The env filter sets the author and committer address of each commit to the personal address of the user.
2. **The patterns stay outside the repo.** The pattern file lives in `~/.cache/dinov3-mlx/release/`. The script contains no private string.
3. **Cited ids follow the rewrite.** In `*.html` and `*.md` files the tree filter replaces each token of 7 to 12 hex digits that names a commit of the old history with the first 7 digits of its new id. Each doc cites only older commits, so the new id exists when the filter runs.
4. **A check stops a bad rewrite.** The script scans all blobs, messages, author and committer fields for the patterns and for local home folder paths, checks the commit count, and resolves each cited id to a commit with the same subject (AC15, AC16). It stops with an error when a check fails.
5. **A new working folder.** Work after the rewrite happens in the new clone `dinov3-mlx`, with the remote of the public repo. The old folder stays as a private archive without a remote.

Why:

- It is the only option with no private string in the public history and valid commit-id citations. The spec loop stays readable on GitHub: each step has its explore, plan, build and verify commits.

We considered:

- **Push the history as it is** (D7 A). Out: the private strings and the company address would become public.
- **One new root commit** (D7 C). Simple, but the 130 citations would point to a history that is not public.

Consequences:

- All commit ids changed. A commit id from before the rewrite is valid only in the private archive.
- A private string that the pattern file does not know stays in the history. The scan covers only the known patterns and home folder paths.
- The ignored files (`CLAUDE.local.md`, `tests/fixtures/generated/`) were copied into the new clone by hand.
- Commits after the rewrite are scanned by T10 and the release checks, not by the rewrite script.

Status: accepted on 2026-11-09 and 2026-11-10; written at the Step 7 Close on 2026-11-14.
