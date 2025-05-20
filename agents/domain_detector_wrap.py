############################################################
# 0. Imports & GPT‑4 model                                 #
############################################################
import json
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv
from .utils import serialize_memory 

# Load environment variables from .env file
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, START, END

MAX_CYCLES = 5

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

llm = ChatOpenAI(
    model_name="gpt-4o", 
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}},
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

############################################################
# 1. LLM‑assisted DataProfiler                             #
############################################################
PROFILE_PROMPT = PromptTemplate(
    input_variables=["raw_preview"],
    template=(
        "You are a data–profiling assistant. Carefully inspect the raw preview of a CSV table and think step‑by‑step to extract *structural facts* that an LLM can later use.\n"
        "--- RAW PREVIEW START ---\n"
        "{raw_preview}\n"
        "--- RAW PREVIEW END ---\n\n"
        "**Think step by step**:\n"
        "1. Count how many rows and columns.\n"
        "2. List every column with an *inferred type* (numeric / categorical / datetime / text).\n"
        "3. For each column give: • three example values  • min / max (if numeric) • detected unit symbol (currency, %, kWh, etc.).\n"
        "4. Detect *functional dependencies* – e.g. if Revenue − Cost ≈ Profit.\n"
        "5. Detect *hierarchical or time‑series structure*: quarter columns, year‑month fields, category → sub‑category columns.\n"
        "6. Identify candidate *primary keys* (columns where all values are unique) and possible *foreign‑key* pairs.\n"
        "7. Note any columns/rows that look like totals or subtotals.\n\n"
        "After reasoning, output a JSON object with keys:\n"
        "  rows, cols, columns (list with {{name, type, examples, unit, min, max}}),\n"
        "  formulas (list of strings),\n"
        "  hierarchy (free text),\n"
        "  time_series (true/false),\n"
        "  candidate_pk (list of column names),\n"
        "  possible_fk (list of tuple strings),\n"
        "  subtotal_cols (list), subtotal_rows (true/false)\n"
        "Use double quotes for all JSON keys. Do **not** wrap the JSON in markdown."
    )
)
profile_chain = PROFILE_PROMPT | llm

def _raw_stats(df: pd.DataFrame, n: int = 5) -> Dict[str, Any]:
    meta = {"n_rows": len(df), "n_cols": df.shape[1], "columns": {}}
    for c in df.columns:
        s = df[c]
        meta["columns"][c] = {
            "dtype": str(s.dtype),
            "unique_ratio": round(s.nunique() / len(s), 4),
            "sample": s.head(n).astype(str).tolist(),
        }
        if pd.api.types.is_numeric_dtype(s):
            meta["columns"][c].update({
                "min": s.min(), "max": s.max(), "mean": round(s.mean(), 4), "std": round(s.std(), 4)
            })
    return meta

def build_profile(csv_path: str) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    raw = _raw_stats(df)
    llm_response = profile_chain.invoke({"raw_preview": json.dumps(raw, cls=NumpyEncoder)})
    # Extract content from AIMessage
    if hasattr(llm_response, 'content'):
        llm_enriched = llm_response.content
    else:
        llm_enriched = str(llm_response)
    return {"raw": raw, **json.loads(llm_enriched)}

############################################################
# 2. Prompt templates                                      #
############################################################
DOMAIN_PROMPT = PromptTemplate(
    input_variables=["profile", "memory"],
    template=(
        "Dataset profile (JSON):\n{profile}\n"
        "{memory}\n"
        "Determine the most precise domain / industry label (Wikipedia terminology).\n"
        "Return JSON response in this format: {{ 'domain':<string>, 'definition':<one sentence>, 'wiki_url':<url|''> }}"
    )
)
domain_chain = DOMAIN_PROMPT | llm

CONCEPT_PROMPT = PromptTemplate(
    input_variables=["profile", "domain_info", "memory"],
    template=(
        "Profile: {profile}\nDomain: {domain_info}\n"
        "{memory}\n"
        "List 4‑6 concept names (English) that should be analysed for this domain.\n"
        "Return the list in JSON format."
    )
)
concept_chain = CONCEPT_PROMPT | llm

ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["profile", "domain_info", "concepts", "memory"],
    template=(
        "Profile: {profile}\nDomain: {domain_info}\nCore concepts: {concepts}\n"
        "{memory}\n"
        "Produce a JSON response exactly in this shape:\n"
        "{{ 'domain': <string>, 'core_concepts': [...], 'analysis': {{ 'descriptive':<paragraph>, 'predictive':<paragraph>, 'domain_related':<paragraph> }} }}\n"
        "Ensure strong **insightfulness** and **novelty**—surface hidden patterns, non‑obvious relationships, or actionable hypotheses."
    )
)
analysis_chain = ANALYSIS_PROMPT | llm

# Evaluator prompt – separate criteria groups
EVAL_PROMPT = PromptTemplate(
    input_variables=["domain_info", "concepts", "analysis", "profile"],
    template=(
        "You are an evaluation agent.\n\n"
        "**Part A – Domain & Concepts**\n"
        "• correctness : is the domain label factually accurate for this dataset?\n"
        "• relevance   : do the concepts correspond to real columns / metrics present?\n"
        "• coverage    : do the concepts cover the major elements of the table?\n\n"
        "**Part B – Analysis JSON**\n"
        "• insightfulness : does the analysis provide meaningful, actionable understanding (combines relevance, uniqueness, actionability)?\n"
        "• novelty        : does it reveal non‑obvious or deeper patterns beyond simple column descriptions?\n\n"
        "Return a JSON response exactly in this format:\n"
        "{{ 'reason':<brief text>,\n  'scores': {{ 'correctness':#, 'relevance':#, 'coverage':#, 'insightfulness':#, 'novelty':# }},\n  'domain_ok': <bool correctness==4>,\n  'concepts_ok': <bool relevance>=3 and coverage>=3> }}"
    )
)
eval_chain = EVAL_PROMPT | llm

REFLECT_PROMPT = PromptTemplate(
    input_variables=["evaluation", "memory"],
    template=(
        "Evaluation JSON: {evaluation}\n"
        "{memory}\n"
        "For every dimension with score ≤3, write ONE bullet‑point self‑critique:\n"
        "  – state what was missing or wrong, and how to improve (e.g. 'Consider column YYYY …').\n"
        "Return ≤5 bullets only, in a valid JSON list format."
    )
)
reflect_chain = REFLECT_PROMPT | llm

############################################################
# 3. Graph node functions                                  #
############################################################

# Helper function to extract content from LLM response
def _extract_content(response):
    if hasattr(response, 'content'):
        return response.content
    return str(response)

def domain_node(state):
    print("domain_node", state)
    """
    Determine the dataset's domain label.
    Inputs  : state["profile"], state["memory"]
    Outputs : domain_info  (dict), profile (unchanged)
    """
    # 0. Fast-path if the caller says our label is already final
    if state.get("domain_fixed"):
        return {**state, **{
            "profile"    : state["profile"],
            "domain_info": state.get("domain_info", {}),
            "memory"     : state.get("memory", "[]"),
            "iteration"  : state.get("iteration", 0)
        }}

    # 1. Always supply memory as a JSON list string – even if empty
    memory_json = serialize_memory(state.get("memory"))

    # 2. Ensure profile is a JSON string for the prompt
    profile_json = (
        json.dumps(state["profile"], cls=NumpyEncoder)
        if isinstance(state["profile"], dict)
        else state["profile"]
    )

    # 3. Call the LLM
    response = domain_chain.invoke({
        "profile": profile_json,
        "memory" : memory_json
    })
    out = _extract_content(response)

    # 4. Parse & validate
    try:
        domain_info = json.loads(out)
        assert isinstance(domain_info, dict)
    except (json.JSONDecodeError, AssertionError):
        domain_info = {
            "domain"    : "Unknown",
            "definition": "Unable to determine domain from the data",
            "wiki_url"  : ""
        }

    # guarantee required keys
    for k in ("domain", "definition", "wiki_url"):
        domain_info.setdefault(k, "")

    # 5. Return updated state
    return {**state, **{
        "profile"    : state["profile"],
        "domain_info": domain_info,
        "memory"     : state.get("memory", "[]"),
        "iteration"  : state.get("iteration", 0)
    }}

def concept_node(state):
    print("concept_node", state)
    """
    Generate 4-6 core concepts for the detected domain.
    Inputs : profile, domain_info, memory
    Output : concepts (Python list)
    """
    memory_json = serialize_memory(state.get("memory"))

    response = concept_chain.invoke({
        "profile"    : json.dumps(state["profile"], cls=NumpyEncoder)
                       if isinstance(state["profile"], dict)
                       else state["profile"],
        "domain_info": json.dumps(state["domain_info"]),
        "memory"     : memory_json
    })

    out = _extract_content(response)

    try:
        concepts = json.loads(out)
        if not isinstance(concepts, list):
            concepts = [concepts]
    except json.JSONDecodeError:
        print("Warning: concept_node received invalid JSON; defaulting to []")
        concepts = []

    return {**state, **{
        "profile"    : state["profile"],
        "domain_info": state["domain_info"],
        "concepts"   : concepts,
        "memory"     : state.get("memory", "[]"),
        "iteration"  : state.get("iteration", 0)
    }}


def analysis_node(state):
    print("analysis_node", state)
    """
    Produce descriptive / predictive / domain-related analysis paragraphs.
    Inputs : profile, domain_info, concepts, memory
    Output : analysis (dict)
    """
    memory_json = serialize_memory(state.get("memory"))

    response = analysis_chain.invoke({
        "profile"    : json.dumps(state["profile"], cls=NumpyEncoder)
                       if isinstance(state["profile"], dict)
                       else state["profile"],
        "domain_info": json.dumps(state["domain_info"]),
        "concepts"   : json.dumps(state["concepts"]),
        "memory"     : memory_json
    })

    out = _extract_content(response)

    try:
        analysis = json.loads(out)
        assert isinstance(analysis, dict)
    except (json.JSONDecodeError, AssertionError):
        print("Warning: analysis_node received invalid JSON; inserting stub.")
        analysis = {
            "domain": state["domain_info"].get("domain", "Unknown"),
            "core_concepts": state["concepts"],
            "analysis": {
                "descriptive"    : "Analysis failed",
                "predictive"     : "Analysis failed",
                "domain_related" : "Analysis failed"
            }
        }

    return {**state, **{
        "profile"    : state["profile"],
        "domain_info": state["domain_info"],
        "concepts"   : state["concepts"],
        "analysis"   : analysis,
        "memory"     : state.get("memory", "[]"),
        "iteration"  : state.get("iteration", 0)
    }}


def eval_node(state):
    print("eval_node", state)
    """
    Evaluation node: Evaluates quality of domain, concepts, and analysis.
    Input: State with all previous outputs
    Output: State with evaluation results
    """
    try:
        # Ensure all inputs are properly serialized
        domain_info = state["domain_info"]
        if isinstance(domain_info, dict):
            domain_info = json.dumps(domain_info)
        
        # Handle concepts (could be list or string)
        concepts = state["concepts"]
        if isinstance(concepts, (list, dict)):
            concepts = json.dumps(concepts)
        
        # Handle analysis (could be dict or string)
        analysis = state["analysis"]
        if isinstance(analysis, dict):
            analysis = json.dumps(analysis)
        
        # Ensure profile is properly serialized
        profile = state["profile"]
        if isinstance(profile, dict):
            profile = json.dumps(profile, cls=NumpyEncoder)
        
        response = eval_chain.invoke({
            "domain_info": domain_info,
            "concepts": concepts,
            "analysis": analysis,
            "profile": profile
        })
        
        out = _extract_content(response)
        ev = json.loads(out)
        
        # Validate evaluation structure
        required_keys = ["reason", "scores", "domain_ok", "concepts_ok"]
        for key in required_keys:
            if key not in ev:
                raise ValueError(f"Missing required key in evaluation: {key}")
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Error in eval_node: {str(e)}")
        ev = {
            "reason": "Evaluation failed",
            "scores": {
                "correctness": 2,
                "relevance": 2,
                "coverage": 2,
                "insightfulness": 2,
                "novelty": 2
            },
            "domain_ok": False,
            "concepts_ok": False
        }
    
    history = state.get("history", [])
    print("history before append", history)

    history.append({
        "iteration": state.get("iteration", 0),
        "domain"   : json.loads(domain_info)["domain"] if isinstance(domain_info, str) else domain_info.get("domain", "Unknown"),
        "scores"   : ev["scores"],
        "analysis_head": (
            json.loads(analysis)["analysis"]["descriptive"][:120]
            if isinstance(analysis, str) else
            (analysis.get("analysis", {}).get("descriptive","")[:120])
        )
    })
    state["history"] = history
    print("history", state["iteration"], "*****", state["history"])
    
    return {**state, **{
        "profile": state["profile"],
        "domain_info": state["domain_info"],
        "concepts": state["concepts"],
        "analysis": state["analysis"],
        "evaluation": ev["reason"],
        "scores": ev["scores"],
        "domain_ok": ev["domain_ok"],
        "concepts_ok": ev["concepts_ok"],
        "memory": state.get("memory", "None"),
        "history": state["history"],
        "iteration": state.get("iteration", 0)
    }}

def reflect_node(state):
    print("reflect_node", state)
    """
    Reflection node: Generates improvement suggestions based on evaluation.
    Input: State with evaluation results
    Output: State with updated memory and iteration count
    """
    # Build evaluation payload for reflector
    eval_payload = {
        "reason": state["evaluation"],
        "scores": state["scores"],
        "domain_ok": state["domain_ok"],
        "concepts_ok": state["concepts_ok"]
    }
    
    # Get previous memory and parse it if it exists
    memory_text = state.get("memory", "None")
    if memory_text != "None":
        if isinstance(memory_text, str):
            try:
                previous_memory = json.loads(memory_text)
                if not isinstance(previous_memory, list):
                    previous_memory = [previous_memory]
            except json.JSONDecodeError:
                previous_memory = []
        elif isinstance(memory_text, list):
            previous_memory = memory_text
        elif isinstance(memory_text, dict):
            previous_memory = [memory_text]
        else:
            previous_memory = []
    else:
        previous_memory = []
    
    # Determine which aspects need reflection based on scores
    needs_reflection = []
    scores = state["scores"]
    
    # Check domain and concept metrics
    if scores.get("correctness", 0) < 4:
        needs_reflection.append("domain")
    if scores.get("relevance", 0) < 4 or scores.get("coverage", 0) < 4:
        needs_reflection.append("concepts")
    if scores.get("insightfulness", 0) < 4 or scores.get("novelty", 0) < 4:
        needs_reflection.append("analysis")
    
    # Only reflect if there are aspects that need improvement
    if needs_reflection:
        # Format previous reflections for the prompt
        if previous_memory:
            memory_text = "Previous reflections:\n" + "\n".join([f"- {ref}" for ref in previous_memory])
        else:
            memory_text = ""
        
        try:
            # Ensure evaluation payload is properly serialized
            eval_json = json.dumps(eval_payload)
            
            response = reflect_chain.invoke({
                "evaluation": eval_json,
                "memory": memory_text
            })
            
            out = _extract_content(response)
            new_reflections = json.loads(out)
            
            # Ensure new_reflections is a list
            if not isinstance(new_reflections, list):
                new_reflections = [new_reflections]
            
            # Add context to each reflection
            for i, reflection in enumerate(new_reflections):
                if isinstance(reflection, str):
                    # Add aspect context to the reflection
                    aspect = needs_reflection[i % len(needs_reflection)]
                    new_reflections[i] = f"[{aspect.upper()}] {reflection}"
            
            # Combine with previous reflections
            combined_reflections = previous_memory + new_reflections
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Error in reflect_node: {str(e)}")
            combined_reflections = previous_memory
    else:
        combined_reflections = previous_memory
    
    # Increment iteration only here
    iteration = state.get("iteration", 0) + 1
    
    # Format memory for next iteration
    formatted_memory = json.dumps(combined_reflections)
    
    return {**state, **{
        "profile": state["profile"],
        "domain_info": state["domain_info"],
        "concepts": state["concepts"],
        "analysis": state["analysis"],
        "evaluation": state["evaluation"],
        "scores": state["scores"],
        "domain_ok": state["domain_ok"],
        "concepts_ok": state["concepts_ok"],
        "memory": formatted_memory,
        "iteration": iteration
    }}

############################################################
# 4. Build state graph - COMPLETELY REVISED                #
############################################################

def success(scores: Dict[str, int]) -> bool:
    """
    Determine if the analysis has reached a successful state.
    
    Args:
        scores: Dictionary of evaluation scores
        
    Returns:
        bool: True if the analysis meets success criteria
    """
    # If scores is empty or None, we can't determine success
    if not scores:
        return False
        
    # Domain and concept metrics
    domain_metrics = {
        'correctness': scores.get('correctness', 0),
        'relevance': scores.get('relevance', 0),
        'coverage': scores.get('coverage', 0)
    }
    
    # Analysis metrics
    analysis_metrics = {
        'insightfulness': scores.get('insightfulness', 0),
        'novelty': scores.get('novelty', 0)
    }
    
    # Domain and concept success criteria - RELAXED FURTHER
    domain_concept_success = (
        domain_metrics['correctness'] >= 4 and  # Domain must be correct (relaxed)
        domain_metrics['relevance'] >= 4 and    # Concepts must be relevant
        domain_metrics['coverage'] >= 4         # Concepts must cover major elements
    )
    
    # Analysis success criteria
    analysis_success = (
        analysis_metrics['insightfulness'] >= 4 and  # Must provide meaningful insights
        analysis_metrics['novelty'] >= 4             # Must reveal non-obvious patterns
    )
    
    return domain_concept_success and analysis_success

def decide_next(state: Dict[str, Any]) -> str:
    # Increment iteration at every decision point
    state["iteration"] = state.get("iteration", 0) + 1
    iteration = state["iteration"]
    scores = state.get("scores", {})
    
    # Print debug info
    print(f"Decision point – iteration {iteration}, scores: {scores}")
    
    # 1. Check hard cap on cycles first - MOST IMPORTANT ESCAPE
    if iteration >= state.get("max_cycles", MAX_CYCLES):
        print(f"🛑 Reached maximum cycles ({state.get('max_cycles', MAX_CYCLES)}), ending execution.")
        return END
    
    # 2. Check for empty scores - indicates initial run or error
    if not scores:
        print("⚠️ No scores available, defaulting to domain node.")
        return "domain"
    
    # 3. Check for success condition
    if success(scores):
        print("✅ Success criteria met, ending execution.")
        return END
    
    # 4. CRITICAL: Check for plateau (no improvement)
    prev_scores = state.get("previous_scores")
    if prev_scores is not None:
        improved = False
        for k, v in scores.items():
            prev_v = prev_scores.get(k, 0)
            if v > prev_v:
                improved = True
                break
        if not improved:
            print("🛑 No improvement detected, ending execution.")
            return END
    # Update previous scores for next iteration
    state["previous_scores"] = scores.copy()
    
    # 5. Route based on what needs improvement
    domain_ok = state.get("domain_ok", False)
    concepts_ok = state.get("concepts_ok", False)
    
    if not domain_ok:
        print(f"🔄 Domain needs improvement (iteration {iteration})")
        return "domain"
    elif not concepts_ok:
        print(f"🔄 Concepts need improvement (iteration {iteration})")
        return "concept"
    else:
        print(f"🔄 Analysis needs improvement (iteration {iteration})")
        return "reflect"

def build_graph(max_cycles: int = MAX_CYCLES):
    """
    Build the analysis state graph with improved anti-recursion measures.
    
    Args:
        max_cycles: Maximum number of improvement cycles
        
    Returns:
        Compiled StateGraph
    """
    # Create a new graph
    builder = StateGraph(dict)
    
    # Add the processing nodes
    builder.add_node("domain", domain_node)
    builder.add_node("concept", concept_node)
    builder.add_node("analysis", analysis_node)
    builder.add_node("eval", eval_node)
    builder.add_node("reflect", reflect_node)
    
    # Set up the standard flow
    builder.add_edge(START, "domain")
    builder.add_edge("domain", "concept")
    builder.add_edge("concept", "analysis")
    builder.add_edge("analysis", "eval")
    
    # IMPORTANT: Connect every node to a next step to prevent dead ends
    builder.add_edge("reflect", "domain")  # Reflect leads back to domain
    
    # Set up conditional branching with explicit edge mapping
    builder.add_conditional_edges(
        "eval",
        decide_next,
        {
            "domain": "domain",
            "concept": "concept",
            "reflect": "reflect",
            END: END
        }
    )
    
    return builder.compile()

############################################################
# 5. Create agent function - COMPLETELY REVISED            #
############################################################
def run_domain_detector(csv_path: str, max_cycles: int = MAX_CYCLES) -> Dict[str, Any]:
    """
    Run the data profiling agent on the specified CSV file with improved error handling.
    
    Args:
        csv_path: Path to the CSV file to analyze
        max_cycles: Maximum number of improvement cycles
        
    Returns:
        Dictionary containing the analysis results
    """
    try:
        # Read the CSV file to verify it exists and is valid
        df = pd.read_csv(csv_path)
        print(f"Successfully read CSV: {len(df)} rows, {len(df.columns)} columns")
        
        # Build profile
        profile = build_profile(csv_path)
        print("Data profile built successfully")
        
        # Create the agent graph
        graph = build_graph(max_cycles=max_cycles)
        
        # Set up initial state with proper initialization
        initial_state = {
            "profile": profile,
            "memory": "[]",  # Initialize as empty JSON array string
            "iteration": 0,
            "max_cycles": max_cycles,  # Add max_cycles to state
            "domain_fixed": False,
            "previous_scores": {},  # Initialize with empty dict instead of None
            "history": []  # Initialize history as empty list
        }

        initial_state["history"].append({
        "iteration": initial_state.get("iteration", 0),
        "domain": initial_state.get("profile", {}).get("domain", "Unknown"),
        "scores": {},
        "analysis_head": ""
        })
        
        # Run the agent with timeout protection
        print(f"Starting analysis with max_cycles={max_cycles}")
        result = graph.invoke(initial_state)
        print("result history",  result["history"])
        
        # Ensure we have at least minimal structure for return value
        if "analysis" not in result:
            result["analysis"] = {
                "domain": "Unknown",
                "core_concepts": [],
                "analysis": {
                    "descriptive": "Analysis incomplete",
                    "predictive": "Analysis incomplete", 
                    "domain_related": "Analysis incomplete"
                }
            }
            
        return result
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        # Return structured error result
        return {
            "error": str(e),
            "analysis": {
                "domain": "Error",
                "core_concepts": [],
                "analysis": {
                    "descriptive": f"Analysis failed: {str(e)}",
                    "predictive": "Analysis failed",
                    "domain_related": "Analysis failed"
                }
            },
            "history": []
        }