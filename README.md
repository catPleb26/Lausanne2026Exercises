# Lausanne 2026 Exercises

Jupyter notebooks for the 37th International Summer School of the Swiss Association of Actuaries, Lausanne, 10–14 August 2026.

## Run in Binder

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/catPleb26/Lausanne2026Exercises/HEAD)

The first launch after a repository update may take several minutes while Binder builds the environment. Later launches reuse the cached environment when possible.

| Notebook | Open directly in Binder |
| --- | --- |
| Day 1 exercises | [Open Day 1](https://mybinder.org/v2/gh/catPleb26/Lausanne2026Exercises/HEAD?urlpath=lab/tree/lausanne_day_1_TODO.ipynb) |
| Day 2 exercises | [Open Day 2](https://mybinder.org/v2/gh/catPleb26/Lausanne2026Exercises/HEAD?urlpath=lab/tree/lausanne_day_2_TODO.ipynb) |
| Day 3 exercises | [Open Day 3](https://mybinder.org/v2/gh/catPleb26/Lausanne2026Exercises/HEAD?urlpath=lab/tree/lausanne_day_3_TODO.ipynb) |

The Day 1 notebook contains `...` placeholders that participants are expected to complete. Run its cells from top to bottom because later exercises reuse functions defined earlier in the notebook and in `common.py`.

Binder sessions are temporary. Before closing a session, download changed notebooks through **File → Save and Export Notebook As → Notebook (`.ipynb`)**. Changes made in Binder are not written back to this GitHub repository.

## Repository contents

- `lausanne_day_1_TODO.ipynb`, `lausanne_day_2_TODO.ipynb`, `lausanne_day_3_TODO.ipynb`: participant notebooks
- `common.py`: shared Python functions used by the notebooks
- `data/`: workshop data
- `requirements.txt` and `runtime.txt`: Binder environment configuration
- `notes.txt`: optional local setup instructions

## Run locally

See `notes.txt` for Windows, macOS, and Linux commands. Python 3.12 is recommended.
