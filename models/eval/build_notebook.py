"""Build models/eval/spall_eval.ipynb from cells.py.

Usage (repo root):  .venv/bin/python -m models.eval.build_notebook
"""

from pathlib import Path

import nbformat as nbf

from .cells import CELLS


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if kind == "md" else nbf.v4.new_code_cell(s) for kind, s in CELLS]
    p = Path(__file__).resolve().parent / "spall_eval.ipynb"
    nbf.write(nb, p)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
