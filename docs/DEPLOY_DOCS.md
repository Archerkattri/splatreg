# Deploying the docs site

The site is mkdocs-material + mkdocstrings; sources live in `docs_site/`, config in
`mkdocs.yml`. There is **no CI** in this repo by design — build and deploy are manual.

## Build + preview locally

```bash
pip install -e ".[docs]"        # mkdocs-material + mkdocstrings[python]
mkdocs serve                    # live preview at http://127.0.0.1:8000
mkdocs build --strict           # the pre-deploy gate (also part of docs/RELEASING.md)
```

## Deploy to GitHub Pages (one command)

```bash
mkdocs gh-deploy --force
```

That builds the site and pushes it to the `gh-pages` branch of `origin`. First time only:
in GitHub → repo **Settings → Pages**, set *Source* to *Deploy from a branch*, branch
`gh-pages` / root. The site then serves at <https://archerkattri.github.io/splatreg/>.

Re-run `mkdocs gh-deploy --force` whenever the docs change. Nothing else to set up.
