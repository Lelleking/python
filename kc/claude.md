# CLAUDE.md

## Project overview
- Flask-app för amyloidkurvor.
- Entry point: `Koder/app.py` (körs med `python app.py`).
- Logik uppdelad i moduler:
  - `config.py` (paths, constants)
  - `state.py` (shared in-memory state)
  - `db.py` (SQLite + saved runs)
  - `ml_models.py` (modell-laddning och prediktion)
  - `data_utils.py` (filparsing, feature-prep)
  - `plot_utils.py` (alla matplotlib-plots)
  - `routes/` (Flask blueprints: `main.py`, `auth.py`, `runs.py`,
    `folders.py`, `halftimes.py`, `sigmoid.py`,
    `aggregation.py`, `event_ai.py`, `plots.py`)

## How to run
- Kör alltid från `kc/`:
  - `cd kc`
  - `python Koder/app.py`

## Rules for Claude Code
- Läs i första hand:
  - `CLAUDE.md`
  - `Koder/app.py`
  - relevanta filer i `Koder/routes/` och `Koder/*_utils.py`
- Undvik att scanna alla csv/txt-filer i rotmapparna (`erik/`, `labs/`, etc)
  om inte jag uttryckligen ber om det.
- När du refaktorerar:
  - behåll `Koder/app.py` som tunn entry (skapa app, registrera blueprints).
  - lägg DB-funktioner i `db.py`,
    ML-funktioner i `ml_models.py`,
    data-hantering i `data_utils.py`,
    plotkod i `plot_utils.py`,
    endpoints i rätt `routes/*.py`.

## When unsure
- Fråga först vilka 1–3 filer du får läsa, i stället för att scanna hela projektet.

## Git — auto-commit after every major change
- After completing any significant set of changes (new feature, bug fix, refactor),
  always run a git commit automatically without being asked.
- Use `git add -A` then commit with a short descriptive message in English.
- Never push unless the user explicitly asks to push.
- Commit message format: one concise line describing what changed, e.g.
  "add plate overview modal and chromatic selection"
