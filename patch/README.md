# How to create a patch file

For a Git repository, you can create a patch file using the `git format-patch` command. Here are the steps to create a patch file:

1. `git log` - Identify the commit(s) you want to include in the patch file and the codebase `base` commit ID.
2. `git reset --soft <base-commit-id>` - Reset the branch to the base commit while keeping the changes staged.
3. `git merge --squash <feature-branch>` - Squash the changes from the feature branch into a single commit.
4. `git commit -m "Your commit message"` - Create a new commit with the squashed changes.
5. `git format-patch -1 HEAD` - Generate a patch file for the latest commit. This will create a `.patch` file in your current directory.
6. Change the file name to `<base-commit-id>.patch` for clarity.

# How to apply a patch

Use the shared script with the patch subdirectory name:

```bash
bash patch/apply_patch.sh vllm
bash patch/apply_patch.sh verl --list
bash patch/apply_patch.sh lm_cache --force dfc914cb.patch
```

By default, the script selects a patch matching the installed distribution's
Git commit, then its version, and finally the most recently modified patch.
Run `bash patch/apply_patch.sh --help` for all options.

To support a new library, add its patch subdirectory and installed Python
distribution name to `register_libraries` in `patch/apply_patch.sh`:

```bash
register_library "patch_subdirectory" "distribution-name"
```
