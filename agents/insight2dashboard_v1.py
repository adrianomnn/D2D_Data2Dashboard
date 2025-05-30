#!/usr/bin/env python
"""
Analysis Generator (schema‑aware, notebook‑friendly)
---------------------------------------------------

Call from a notebook:

    from generate_and_run import generate_analysis
    thoughts = generate_analysis(
        csv_path='finance_data.csv',
        insight_json_path='insight_library.json',
        model='gpt-4o',
        run_code=True
    )

Key upgrades (v3.1):
• Pass **actual dataframe schema** (column names + dtypes) to GPT so it
  references existing columns, avoiding KeyError crashes.
• Instruct GPT to *verify* column presence and wrap each plot section in
  a `try/except KeyError` block that logs a warning instead of failing.
• Allow `run_code=False` to preview thoughts/code without executing.
• **NEW** – read the *entire* CSV (not just `nrows=0`) when deriving the
  schema, ensuring dtypes such as `int64`, `float64`, `object`, etc. are
  inferred more accurately on real data.
"""

from __future__ import annotations

import os
import re
import json
import subprocess
import textwrap
from pathlib import Path
import sys
from typing import Tuple

from dotenv import load_dotenv
import openai
import pandas as pd

# ──────────────────────────────────────────────────────────────────
# Prompt template (includes schema + robustness instructions)
# ──────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """
You are an elite data‑visualisation consultant.

Context:
  • **insight_library** (JSON):
{insight_json}
  • **CSV_SCHEMA** (column → dtype):
{schema_table}
  • **CSV_PATH** – a string pointing to the dataset on disk.

Return **exactly two fenced blocks** in order and nothing else:

1️⃣ Thoughts block  (label it ```thoughts)
   – Identify 3–5 key business questions the data can answer, tying
     each to `core_concepts`. Keep ≤250 words.

2️⃣ Python block  (label it ```python)
   – A self‑contained script **robust to missing columns & type issues**:
       • import pandas as pd, matplotlib.pyplot as plt, Path
       • read dataset via provided CSV_PATH variable (already defined)
       • **When selecting multiple columns, always use list syntax**
         (`df[["col1", "col2"]]`) – **never** a tuple – to avoid
         `ValueError: Cannot subset columns with a tuple …`.
       • one figure per business question; wrap each plot in
         `try/except (KeyError, ValueError, TypeError)` and
         `print()` a warning if a problem occurs instead of stopping.
       • call plt.tight_layout(); show() or save to figures/
   – Use only columns present in CSV_SCHEMA.
"""

# ──────────────────────────────────────────────────────────────────
# Helper – chat and extract blocks
# ──────────────────────────────────────────────────────────────────

def _chat_and_extract(*, insight_json: str, schema_table: str, csv_path: str,
                      model: str, temperature: float) -> Tuple[str, str]:
    user_prompt = PROMPT_TEMPLATE.format(
        insight_json=insight_json,
        schema_table=schema_table,
    )

    system_msg = (
        "Answer strictly with two fenced blocks: first ```thoughts, then ```python."
    )

    rsp = openai.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = rsp.choices[0].message.content

    thoughts_m = re.search(r"```thoughts(.*?)```", content, re.S)
    code_m     = re.search(r"```python(.*?)```", content, re.S)
    if not (thoughts_m and code_m):
        raise ValueError("GPT response missing required fenced blocks.")
    return thoughts_m.group(1).strip(), code_m.group(1).strip()

# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

def generate_analysis(
    csv_path: str | Path,
    insight_json_path: str | Path,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    run_code: bool = True,
    save_dir: str | Path = ".",
) -> str:
    """Generate thoughts & code; optionally execute.

    Returns the design‑rationale (thoughts) string.
    """

    csv_path = Path(csv_path).expanduser().resolve()
    insight_json_path = Path(insight_json_path).expanduser().resolve()
    save_dir = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not insight_json_path.exists():
        raise FileNotFoundError(insight_json_path)

    # --- read insight JSON & dataframe schema --------------------------------
    insight_json_str = insight_json_path.read_text(encoding="utf-8")

    # Read the full CSV to capture real dtypes (helps with int/float/object)
    df_sample = pd.read_csv(csv_path)
    schema_table = "\n".join(f"- {col}: {dtype}" for col, dtype in df_sample.dtypes.items())

    # --- auth ----------------------------------------------------------------
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    # --- chat -----------------------------------------------------------------
    thoughts, code_str = _chat_and_extract(
        insight_json=insight_json_str,
        schema_table=schema_table,
        csv_path=str(csv_path),
        model=model,
        temperature=temperature,
    )

    # --- write artefacts ------------------------------------------------------
    thoughts_file = save_dir / "analysis_thoughts.md"
    code_file     = save_dir / "analysis.py"

    thoughts_file.write_text(thoughts, encoding="utf-8")

    header = textwrap.dedent(
        f"""
        # Auto‑injected by generate_analysis
        from pathlib import Path
        CSV_PATH = Path(r"{csv_path}")
        """
    )
    code_file.write_text(header + "\n" + code_str, encoding="utf-8")

    print(f"🧠  Thoughts saved → {thoughts_file}")

    if run_code:
        print("🚀  Executing generated analysis script…")
        subprocess.run([sys.executable, str(code_file)], check=True)

    return thoughts

# ──────────────────────────────────────────────────────────────────
# CLI fallback
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_and_run.py data.csv insight.json")
        sys.exit(1)

    generate_analysis(sys.argv[1], sys.argv[2])
