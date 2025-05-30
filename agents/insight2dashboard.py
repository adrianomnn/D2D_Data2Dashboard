# == 0. Imports & LLM =========================================================
import os
import json, pandas as pd, nbformat, uuid, textwrap
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


from dotenv import load_dotenv
load_dotenv()

# LLM for JSON-formatted responses
llm = ChatOpenAI(
    model_name="gpt-4o", 
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# LLM for code generation (no JSON format restriction)
code_llm = ChatOpenAI(
    model_name="gpt-4o", 
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# == 1. Chains ================================================================
parse_prompt = PromptTemplate(
    input_variables=["stage1_analysis"],
    template=textwrap.dedent("""
        You are a strict JSON extractor.
        From the INPUT file, return a JSON object with exactly the following fields    :
        {{
          "domain":            string | null,
          "core_concepts":     [string, ...] | [],
          "descriptive":       string | null,
          "predictive":        string | null,
          "domain_related":    string | null
        }}

        If a field is missing, set it to null (or [] for arrays).
        Do NOT wrap the JSON in markdown fences or add commentary.

        INPUT:
        ======
        {stage1_analysis}
    """)
)
parse_chain = parse_prompt | llm

planning_prompt = PromptTemplate(
    input_variables=["domain", "core_concepts", "insights", "columns"],
    template="""
You are designing a {domain} dashboard, including the {core_concepts}.

INPUT INSIGHTS (json): {insights}
DATA COLUMNS: {columns}

◉ Produce 4-6 chart specs **as a JSON list**.
◉ Each chart **must include**:
  - "title": short
  - "category": "descriptive" | "predictive" | "domain_related"
  - "type": one of "bar" | "line" | "hist" | "box" | "scatter"
  - "x": column name (or None for hist)
  - "y": column/agg (or None for pie/hist)
  - "why": one-sentence rationale based on the insights and the domain
  - *If* category == "predictive", also return
      "use_linear_regression": true | false
      (Let GPT-4o judge whether a simple linear model will add value.)

Return **only** the JSON list – no extra text.
"""
)

planning_chain = planning_prompt | llm

code_prompt = PromptTemplate(
    input_variables=["chart"],
    template="""
You are writing a Jupyter Notebook cell.

Dataset `df` is already in memory (a pandas DataFrame).

Chart specification:
{chart}

Write ONE code cell that:
  • Imports matplotlib & seaborn if needed
  • Uses `df` to compute any aggregates (e.g. df['col'].value_counts())
  • If chart["category"] == "predictive" and chart["use_linear_regression"]:
        - Use seaborn.regplot(...) and print R²
  • Otherwise, plot normally using the appropriate chart type
  • Sets a descriptive title and axis labels
  • Includes plt.tight_layout() for proper spacing

Output one ```python-fenced code cell.
"""
)

# Use code_llm instead of llm for code generation
code_chain = code_prompt | code_llm

# == 2. Agent wrapper =========================================================
class Insight2DashboardAgent:
    def __init__(self, llm=llm):
        self.llm = llm

    # --------------------------------------------------------------------- #
    # 1.  ANALYSIS EXTRACTION
    # --------------------------------------------------------------------- #
    def _extract_analysis(self, raw_text: str) -> dict:
        """
        Parse stage-1 analysis into a dict with keys:
        domain · core_concepts · descriptive · predictive · domain_related
        """
        # direct-JSON fast-path
        try:
            parsed = json.loads(raw_text)
            if {"domain", "core_concepts", "analysis"}.issubset(parsed):
                return {
                    "domain":         parsed.get("domain"),
                    "core_concepts":  parsed.get("core_concepts", []),
                    "descriptive":    parsed["analysis"].get("descriptive"),
                    "predictive":     parsed["analysis"].get("predictive"),
                    "domain_related": parsed["analysis"].get("domain_related"),
                }
        except Exception:
            pass

        # LLM fallback
        ai_msg = parse_chain.invoke({"stage1_analysis": raw_text})
        parsed_json = ai_msg.content if hasattr(ai_msg, "content") else ai_msg
        return json.loads(parsed_json)

    # --------------------------------------------------------------------- #
    # 2.  MAIN ENTRYPOINT
    # --------------------------------------------------------------------- #
    def run(self, csv_path: str, analysis_text: str, auto_exec: bool = True):
        import pandas as pd

        # 2-A) Load CSV -----------------------------------------------------
        df = pd.read_csv(csv_path)
        print("▶ Loaded CSV:", csv_path)                                 # <<< checkpoint >>>
        print("    shape:", df.shape, "\n")

        # 2-B) Extract structured analysis ---------------------------------
        analysis_dict = self._extract_analysis(analysis_text)
        print("▶ Parsed analysis dict")                                   # <<< checkpoint >>>
        print(json.dumps(analysis_dict, indent=2, ensure_ascii=False), "\n")

        insights_json = json.dumps(analysis_dict, ensure_ascii=False)

        # 2-C) Chart-planning prompt ---------------------------------------
        ai_msg_plan = planning_chain.invoke({
            "domain": analysis_dict.get("domain", "N/A"),
            "core_concepts": ", ".join(analysis_dict.get("core_concepts", [])) or "no core concepts",
            "insights": insights_json,
            "columns": ", ".join(df.columns),
        })
        plan_raw = ai_msg_plan.content if hasattr(ai_msg_plan, "content") else ai_msg_plan
        
        plan_obj = json.loads(plan_raw)
        
        # ---- normalise to list[dict] -----------------------------------------
        if isinstance(plan_obj, dict):
            # Case: {"charts": [chart1, chart2, ...]} - extract the list from the "charts" key
            if "charts" in plan_obj and isinstance(plan_obj["charts"], list):
                plan = plan_obj["charts"]
            else:
                plan = [plan_obj]                            # single chart → wrap in list
        elif isinstance(plan_obj, list):
            # if elements are still JSON strings, decode each one
            if plan_obj and isinstance(plan_obj[0], str):
                plan = [json.loads(p) for p in plan_obj]
            else:
                plan = plan_obj                          # already list[dict]
        else:
            raise ValueError(f"Planning step returned an unexpected JSON type.")
        
        # ---- sanity-check each item ------------------------------------------
        required = {"title", "category", "type"}
        cleaned  = []
        
        for item in plan:
            # 1) If item is a wrapped dict {"chart": "{...json string...}"}
            if isinstance(item, dict) and len(item) == 1 and isinstance(next(iter(item.values())), str):
                try:
                    inner = json.loads(next(iter(item.values())))
                    item  = inner
                except Exception:
                    pass
            
            # 2) Validate keys
            if not isinstance(item, dict) or not required.issubset(item):
                print(f"⚠  Unexpected chart spec - dumping object for debugging:\n", item, "\n")
                raise ValueError("Planning step did not return dicts with keys "
                                "'title', 'category', 'type'.  See object above.")
            cleaned.append(item)
        
        plan = cleaned  # replace with validated list
            
        print(f"▶ Chart plan ({len(plan)} charts)")      # <<< checkpoint >>>
        for idx, c in enumerate(plan, 1):
            print(f"  {idx}. {c['title']}  –  {c['category']}/{c['type']}")
        print()

        # 2-E) Notebook-code generation (one chart at a time) --------------
        print(f"▶ Generating code for {len(plan)} charts...")
        
        all_code_cells = []
        for i, chart in enumerate(plan, 1):
            print(f"  - Chart {i}: {chart['title']}")
            
            # Generate code for this single chart
            chart_json = json.dumps(chart, ensure_ascii=False)
            try:
                ai_msg_code = code_chain.invoke({"chart": chart_json})
                code_cell = ai_msg_code.content if hasattr(ai_msg_code, "content") else ai_msg_code
                all_code_cells.append(code_cell)
            except Exception as e:
                print(f"    ⚠️ Error generating code for chart {i}: {e}")
                all_code_cells.append(f"```python\n# Error generating code for {chart['title']}: {str(e)}\n```")
                
        # Combine all code cells
        code_cells_md = "\n\n".join(all_code_cells)
        print(f"▶ Generated code for {len(plan)} charts ({len(code_cells_md)} chars)")
        print()

        # 2-F) Display / execute in-notebook -------------------------------
        if auto_exec:
            from IPython.display import display, Markdown
            display(Markdown(code_cells_md))
            
            # Extract and execute the Python code inside the markdown code blocks
            import re
            
            # Extract code blocks using regex to avoid markdown parsing issues
            code_blocks = re.findall(r'```python\n(.*?)```', code_cells_md, re.DOTALL)
            
            cells_to_exec = []
            for code_block in code_blocks:
                # Skip blocks that are error messages
                if not code_block.strip().startswith("# Error generating code"):
                    cells_to_exec.append(code_block.strip())
            
            if cells_to_exec:
                # Join the valid code blocks and execute them
                exec("\n\n".join(cells_to_exec), globals(), locals())
            else:
                print("No valid code cells to execute")
        else:
            return code_cells_md

