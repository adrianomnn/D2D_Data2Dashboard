# agents/formula_retriever_v0.py
import json, re
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

MODEL_NAME   = "gpt-4o"
chat_formula = ChatOpenAI(
    model_name     = MODEL_NAME,
    temperature    = 0,
    max_tokens     = 256,
    openai_api_key = None,
    model_kwargs   = {"response_format": {"type": "json_object"}},
)

# Load prompt from external file
with open("prompts/formula_retriever_prompt.txt") as f:
    PROMPT_TMPL = f.read()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def _ask_formula(term: str, domain: str) -> dict:
    prompt = PROMPT_TMPL.format(term=term, domain=domain)
    raw = chat_formula.invoke(prompt).content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1].strip()
    if raw.lower().startswith("json"):
        raw = raw[4:].lstrip()
    first_brace = raw.find("{")
    last_brace  = raw.rfind("}")
    if first_brace == -1 or last_brace == -1:
        raise ValueError("No JSON object found in model response:\n" + raw)

    json_str = raw[first_brace : last_brace + 1]
    return json.loads(json_str)

def get_formula(jargon_rows: list[dict]) -> list[dict]:
    enriched = []
    for row in jargon_rows:
        term   = row["jargon_term"]
        domain = row["domain"]

        try:
            data = _ask_formula(term, domain)
            row  = {**row, **data}
        except Exception as e:
            print(type(e).__name__, "→", e)
            row = {**row, "formula": None, "variables": {}, "error": str(e)}
        
        enriched.append(row)

    return enriched
