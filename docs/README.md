# PSRL Documentation

Source for the PSRL documentation site, built with [Sphinx](https://www.sphinx-doc.org/)
and published on [Read the Docs](https://psrl.readthedocs.io/en/latest/).

This README is a guide for people editing the docs. It is excluded from the built
site.

---

## Build locally

From the repository root:

```bash
# 1. Install the documentation toolchain
python -m pip install -r docs/requirements.txt

# 2. Build the HTML site
cd docs
make html

# 3. Preview it
python -m http.server -d _build/html 8000
```

Then open <http://localhost:8000>.

Building the docs does not import PSRL itself, so a full PSRL environment (vLLM, Ray,
Megatron) is not required. The packages in `requirements.txt` are enough and install
cleanly into a bare virtualenv.

For an auto-reloading dev server that rebuilds on save:

```bash
python -m pip install sphinx-autobuild
cd docs
make livehtml
```

`make clean` removes `_build/`, and `make help` lists every target Sphinx offers.

To catch broken references the way CI does, promote warnings to errors:

```bash
cd docs
make html SPHINXOPTS="-W --keep-going"
```

---

## Layout

```
docs/
├── conf.py            # Sphinx configuration
├── Makefile           # make html / livehtml / clean
├── requirements.txt   # Sphinx, theme, and extensions
├── README.md          # This guide (not part of the site)
├── index.md           # Landing page and top-level toctree
├── _static/           # CSS, JS, and figures (SVG, PNG, MP4)
├── _templates/        # Theme template overrides
├── overview/          # What PSRL is and how it fits together
├── tutorial/          # Installation, quickstart, configuration reference
├── design/            # Architecture and subsystem deep dives
└── examples/          # End-to-end recipes (RLVR, agentic RL, reward models)
```

`.readthedocs.yaml` lives at the repository root, where Read the Docs expects it.

---

## Editing

### Add a page

1. Create the `.md` file in the right section, for example `docs/design/new_feature.md`.
2. Register it in that section's `index.md` toctree, otherwise Sphinx warns that the
   page is orphaned:

   ````markdown
   ```{toctree}
   :maxdepth: 1

   architecture
   parameter_server
   new_feature
   ```
   ````

3. Rebuild and check for warnings.

### Markdown flavor

Pages are MyST Markdown. Enabled extensions include `colon_fence`, `deflist`,
`dollarmath`, `amsmath`, `attrs_inline`, `substitution`, and `tasklist`, so
definition lists, `$...$` math, and `:::`-fenced directives all work.

Cross-reference other pages with `` {doc}`../design/architecture` `` instead of a raw
relative link, so Sphinx validates the target at build time.

### Diagrams

Mermaid renders client-side:

````markdown
```{mermaid}
graph LR
    A[Train Worker] -->|push| B[Parameter Server]
    B -->|pull| C[Rollout Instance]
```
````

### Figures

Put figures in `docs/_static/img/` and reference them with a root-relative path:

````markdown
```{figure} /_static/img/overview.svg
```
````

Prefer SVG. To convert a PDF figure, use `pdf2svg in.pdf out.svg` or
`inkscape in.pdf --export-filename=out.svg`.

The resource elasticity overview keeps its PDF source at
`docs/_static/img/resource_elasticity_overview.pdf`. Regenerate its SVG from the
repository root with:

```bash
python -m pip install PyMuPDF
python examples/elastic/elastic_analysis/render_elastic_overview.py
```

The repository `.gitignore` excludes `**/*.png` as a runtime artifact, with an
explicit exception for `docs/_static/img/*.png`. Keep new raster figures in that
directory so they are not silently dropped from commits.

### Sphinx Design components

```markdown
:::{grid-item-card} Title
:link: target/page
:link-type: doc
Description text.
:::

::::{tab-set}
:::{tab-item} FSDP
FSDP instructions.
:::
:::{tab-item} Megatron
Megatron instructions.
:::
::::

:::{admonition} Title
:class: tip
Content.
:::
```

---

## Publishing

Read the Docs builds from `.readthedocs.yaml` at the repository root, which points at
`docs/conf.py` and installs `docs/requirements.txt` on Python 3.11. Pushing to the
default branch triggers a rebuild.

To validate docs on pull requests:

```yaml
# .github/workflows/docs.yml
name: Docs
on: [pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r docs/requirements.txt
      - run: make -C docs html SPHINXOPTS="-W --keep-going"
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not import extension myst_parser` | `pip install -r docs/requirements.txt` |
| `Theme error: no theme named 'pydata_sphinx_theme'` | Same, the theme ships in `requirements.txt` |
| Mermaid block renders as plain text | Confirm `sphinxcontrib-mermaid` is installed and the fence is `{mermaid}` |
| `document isn't included in any toctree` | Add the page to its section `index.md`, or list it in `exclude_patterns` |
| Figure missing from output | Check the file is under `docs/_static/img/` and the path starts with `/_static/img/` |
| Stale output after edits | `make clean && make html` |
