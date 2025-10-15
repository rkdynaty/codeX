# Resolving the `scripts/analyze_stock.py` merge conflict for PR #3

This guide walks through the exact Git commands you can use to merge [PR #3](https://github.com/rkdynaty/codeX/pull/3)
when your local branch reports a conflict in `scripts/analyze_stock.py`.

## 1. Fetch the pull request locally

```bash
git fetch origin pull/3/head:pr-3
```

This creates (or updates) a local branch called `pr-3` that matches the pull request.

## 2. Check out the branch you want to merge into

Most contributors merge into `main`, but swap in whichever branch you are using:

```bash
git checkout main
git pull
```

Pulling ensures your local branch is up to date before you start the merge.

## 3. Merge the PR branch and resolve conflicts

```bash
git merge pr-3
```

Git will stop and mark conflicts in `scripts/analyze_stock.py`. Open the file in your editor and look for sections bounded by
`<<<<<<<`, `=======`, and `>>>>>>>`. The content between `<<<<<<< HEAD` and `=======` is your current branch. The content between
`=======` and `>>>>>>> pr-3` is from the pull request.

Decide which parts to keep—or combine the changes manually—then delete the conflict markers.

If you want to keep **all changes from the pull request** for that file, you can shortcut with:

```bash
git checkout --theirs -- scripts/analyze_stock.py
```

If you prefer to start from your current branch and selectively copy in bits from the PR, use:

```bash
git checkout --ours -- scripts/analyze_stock.py
```

After choosing a side, open the file, review it, and make any necessary manual adjustments so the final version reflects what you expect.

## 4. Mark the conflict as resolved

```bash
git add scripts/analyze_stock.py
```

Repeat for any other conflicted files. Verify that Git no longer lists conflicts:

```bash
git status
```

## 5. Complete the merge

Once all conflicts are resolved and staged:

```bash
git commit
```

Git will generate a default merge commit message, which you can accept or edit.

## 6. Push the merged branch

```bash
git push origin main
```

(Replace `main` with the appropriate branch if needed.)

## Troubleshooting tips

- Run `git diff` to double-check the resolved file before committing.
- If you make a mistake, you can restart the merge with `git merge --abort` and try again.
- Use `python scripts/analyze_stock.py --help` to confirm the script still runs after the merge.

Following these steps will merge PR #3 while ensuring `scripts/analyze_stock.py` ends up in a consistent state.
