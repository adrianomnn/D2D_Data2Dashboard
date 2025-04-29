# ---------- imports ----------
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END

import json, re
from typing import List, Dict, Any, Optional

# ---------- state ----------
class DomainDetectorState(dict):
    def __init__(
        self,
        columns: List[str],
        column_types: Dict[str, str],
        domains_found: List[Dict[str, Any]] = None,
        current_domain: Optional[str] = None,
        jargon_terms: List[str] = None,
        remaining_columns: List[str] = None,
        iteration: int = 0,
        messages: List[str] = None,
    ):
        super().__init__(
            columns=columns,
            column_types=column_types,
            domains_found=domains_found or [],
            current_domain=current_domain,
            jargon_terms=jargon_terms or [],
            remaining_columns=remaining_columns or list(columns),
            iteration=iteration,
            messages=messages or [],
        )

# ---------- LLM ----------
llm = OpenAI(temperature=0)

# ---------- think ----------
def think(state: DomainDetectorState) -> DomainDetectorState:
    if len(state["domains_found"]) >= 5 or not state["remaining_columns"]:
        return state

    prompt = PromptTemplate.from_template(
        """
        You are a domain expert.
        Remaining columns: {remaining_columns}
        Column types: {column_types}
        Domains already found: {domains_found}
        Identify ONE new domain (single word / short phrase).
        """
    )
    name = LLMChain(llm=llm, prompt=prompt).run(
        remaining_columns=state["remaining_columns"],
        column_types=state["column_types"],
        domains_found=[d["domain"] for d in state["domains_found"]],
    ).strip()

    new = state.copy()
    new["current_domain"] = name
    new["messages"].append(f"Think → {name}")
    return new

# ---------- helper ----------
def search_examples(domain: str, n: int = 5) -> List[str]:
    prompt = PromptTemplate.from_template(
        """
        You are a {domain} specialist.
        List {n} jargon terms in JSON array form.
        """
    )
    raw = (prompt | llm | StrOutputParser()).invoke({"domain": domain, "n": n})
    try:
        return json.loads(raw)[:n]
    except Exception:
        match = re.search(r"\[(.*?)\]", raw, re.S)
        return re.split(r"[\"',\[\]]+", match.group(1))[:n] if match else []

# ---------- act ----------
def act(state: DomainDetectorState) -> DomainDetectorState:
    if not state["current_domain"]:
        new = state.copy(); new["iteration"] += 1; return new

    examples = search_examples(state["current_domain"])

    prompt = PromptTemplate.from_template(
        """
        Domain: {domain}
        Columns: {columns}
        Example jargon: {examples}
        Produce ≥5 NEW jargon terms (JSON array).
        """
    )
    raw = LLMChain(llm=llm, prompt=prompt).run(
        domain=state["current_domain"],
        columns=state["columns"],
        examples=examples,
    )
    try:
        jargon = json.loads(re.search(r"\[.*?\]", raw, re.S).group(0))
    except Exception:
        jargon = []

    new = state.copy()
    new["jargon_terms"] = jargon
    new["messages"].append(f"Act → {jargon}")
    new["iteration"] += 1
    return new

# ---------- reflect ----------
def reflect(state: DomainDetectorState) -> DomainDetectorState:
    if not state["current_domain"] or not state["jargon_terms"]:
        return state

    cols_lower = {c.lower() for c in state["columns"]}
    valid = [t for t in state["jargon_terms"] if t.lower() not in cols_lower]

    new = state.copy()
    if len(valid) >= 3:
        new["domains_found"].append(
            {"domain": state["current_domain"], "jargon_terms": valid[:5]}
        )
        new["messages"].append(f"Reflect ✔ {state['current_domain']}")
        if new["remaining_columns"]:
            new["remaining_columns"].pop(0)
    else:
        new["messages"].append(f"Reflect ✖ {state['current_domain']}")

    new["current_domain"] = None
    new["jargon_terms"] = []
    new["iteration"] += 1
    return new

# ---------- stop rule ----------
def should_end(state: DomainDetectorState) -> str:
    if len(state["domains_found"]) >= 5 or not state["remaining_columns"] or state["iteration"] >= 10:
        return "end"
    return "continue"

# ---------- graph ----------
def build_graph():
    g = StateGraph(DomainDetectorState)
    g.add_node("think", think)
    g.add_node("act", act)
    g.add_node("reflect", reflect)
    g.add_edge("think", "act")
    g.add_edge("act", "reflect")
    g.add_conditional_edges("reflect", should_end, {"continue": "think", "end": END})
    g.set_entry_point("think")
    return g.compile()

# ---------- run ----------
def run_domain_detector(columns, column_types=None):
    column_types = column_types or {c: "text" for c in columns}
    state0 = DomainDetectorState(columns=columns, column_types=column_types)
    return build_graph().invoke(state0, config={"recursion_limit": 50})["domains_found"]

# ---- quick test ----
if __name__ == "__main__":
    cols = ["Quarter", "number_Customers", "Total_Transactions", "Revenue", "Profit"]
    print(json.dumps(run_domain_detector(cols), indent=2))
