# Releasing splatreg

Maintainer checklist for cutting a release. No CI is used in this repo — every
step below is run locally and manually, by design.

## 1. Version bump

Update the version in **three** places (keep them in sync):

- `pyproject.toml` → `[project] version`
- `splatreg/__init__.py` → `__version__`
- `CITATION.cff` → `version` + `date-released`

## 2. Local gate

```bash
python -m pytest tests/ -q          # full suite must pass
mkdocs build --strict               # docs must build clean (pip install -e ".[docs]")
```

## 3. Build + publish to PyPI

```bash
rm -rf dist/
python -m build                     # pip install build
twine upload dist/*                 # pip install twine; uses ~/.pypirc
```

## 4. Tag + GitHub release

```bash
git tag -a v1.0.X -m "splatreg v1.0.X"
git push origin v1.0.X
gh release create v1.0.X --title "splatreg v1.0.X" --notes "..."
```

## 5. Zenodo DOI (one-time setup, then automatic per release)

Zenodo archives each GitHub release and mints a citable DOI. ~30 minutes once:

1. Go to <https://zenodo.org> and **log in with the GitHub account** that owns
   `Archerkattri/splatreg` (Log in → GitHub OAuth).
2. Open <https://zenodo.org/account/settings/github/>, find
   `Archerkattri/splatreg` in the repository list, and flip the toggle **ON**.
   (The repo must be public for Zenodo to see it.)
3. Publish a GitHub release (step 4 above). Zenodo picks it up automatically
   within a few minutes and mints **two** DOIs:
   - a **concept DOI** that always resolves to the latest version — put this
     one in the README badge and in `CITATION.cff`;
   - a **version DOI** for that specific release.
4. Copy the concept DOI into:
   - `CITATION.cff` → uncomment and fill the `doi:` line;
   - `README.md` → add the badge
     `[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)`.
5. (Optional) On the Zenodo record page, hit **Edit** to polish the metadata —
   Zenodo pre-fills title/authors from `CITATION.cff`, so it should already be
   correct.

Subsequent releases need nothing: every new GitHub release gets a new version
DOI under the same concept DOI.

## 6. JOSS (later)

At the ~6-month public mark (~Dec 2026), submit a short JOSS paper
(<https://joss.theoj.org/>). JOSS requires: an OSI license (BSD-3 ✓), a
`paper.md` + `paper.bib`, archived release (the Zenodo DOI above ✓), and a
documented public API (the docs site ✓).
