# Project Contract

## ALWAYS

- Apply first-principles thinking. Do not assume that I always have a clear understanding of what I want or how to achieve it. Stay cautious and start from the fundamental needs and problem. If the motivation or objective is unclear, pause and discuss it with me. 
- When running scripts or inspecting the environment, please activate the conda environment by executing `source env/psrl.sh`. All dependencies and packages are installed within this environment.
- Make architectural decisions for the long term. Do not accept a stopgap that only works for now and is meant to be replaced later.
- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall complexity or improve reliability. Do not reimplement common functionality without a clear reason.

## Coding Guidelines

Three reference files live under `.claude/`:

- **`.claude/coding-style.md`** — formatting rules, naming conventions, docstrings, logging, and annotation markers. Ordered by risk (silent-bug rules first).
- **`.claude/codebase-map.md`** — system architecture, directory tree, configuration hierarchy, quick-lookup indices, and import dependency graphs.
- **`.claude/readme-style.md`** — what belongs in a README and what does not. Read it before writing or editing any user-facing `.md`.

Claude must read and apply these guides when writing or modifying code.

## Documentation

- A README tells a reader what to do and what will bite them. It is not the lab notebook that proves how we learned it.
- Do not paste measured forensics into documentation. Keep a number only if the reader acts on it, and drop the run-specific evidence that merely justifies a past decision. Experiment results belong in a `Results` section or a `FINDINGS.md`, tied to the script that reproduces them.
- Never write history into documentation or comments. `git log` owns what the code used to be.
- Everything here is published. No absolute paths, real hostnames, cluster IPs, or internal mirrors, and every command must run exactly as written.

## Compact Instructions

When compressing, preserve in priority order:

1. Architecture decisions (NEVER summarize)
2. Modified files and their key changes
3. Current verification status (pass/fail)
4. Open TODOs and rollback notes
5. Tool outputs (can delete, keep pass/fail only)
