from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import re
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

# Define the state schema
class DomainDetectorState(dict):
    """State for the domain detector agent."""
    def __init__(self, 
                 columns: List[str],
                 column_types: Dict[str, str],
                 domains_found: List[Dict[str, Any]] = None,
                 current_domain: Optional[str] = None,
                 jargon_terms: List[str] = None,
                 remaining_columns: List[str] = None,
                 iteration: int = 0,
                 messages: List[str] = None):
        
        self.columns = columns
        self.column_types = column_types
        self.domains_found = domains_found or []
        self.current_domain = current_domain
        self.jargon_terms = jargon_terms or []
        self.remaining_columns = remaining_columns or list(columns)
        self.iteration = iteration
        self.messages = messages or []
        
        super().__init__(
            columns=self.columns,
            column_types=self.column_types,
            domains_found=self.domains_found,
            current_domain=self.current_domain,
            jargon_terms=self.jargon_terms,
            remaining_columns=self.remaining_columns,
            iteration=self.iteration,
            messages=self.messages
        )

# Initialize LLM
llm = OpenAI(temperature=0)

# Define the "Think" node - Identify a potential domain
def think(state: DomainDetectorState) -> DomainDetectorState:
    """Identify a potential domain based on remaining columns."""
    
    # Skip if we already have 5 domains or no columns remain
    if len(state["domains_found"]) >= 5 or not state["remaining_columns"]:
        return state
    
    think_template = """
    You are a domain expert tasked with identifying knowledge domains from a set of data columns.
    
    Current columns to analyze: {remaining_columns}
    Column types: {column_types}
    
    Domains already identified: {domains_found}
    
    Based on the remaining columns, identify ONE new knowledge domain that best represents them.
    Consider business, scientific, or technical domains that would use such terminology.
    
    Return only the domain name as a single word or short phrase.
    """
    
    think_prompt = PromptTemplate(
        template=think_template,
        input_variables=["remaining_columns", "column_types", "domains_found"]
    )
    
    think_chain = LLMChain(llm=llm, prompt=think_prompt)
    
    domain = think_chain.run({
        "remaining_columns": state["remaining_columns"],
        "column_types": state["column_types"],
        "domains_found": [d["domain"] for d in state["domains_found"]]
    }).strip()
    
    new_state = state.copy()
    new_state["current_domain"] = domain
    new_state["messages"].append(f"Identified potential domain: {domain}")
    
    return new_state

# Mock function for search_examples
def search_examples(domain: str) -> List[str]:
    """Mock function to simulate fetching jargon hints for a domain."""
    # In a real implementation, this would call an external API or database
    domain_examples = {
        "Finance": ["ROI", "EBITDA", "liquidity", "amortization", "depreciation"],
        "Retail": ["SKU", "inventory turnover", "markdown", "POS", "shrinkage"],
        "E-commerce": ["conversion rate", "cart abandonment", "AOV", "CPC", "CTR"],
        "Sales": ["pipeline", "lead generation", "churn rate", "upselling", "quota"],
        "Marketing": ["CAC", "LTV", "engagement rate", "attribution", "funnel"],
        "Business Intelligence": ["KPI", "dashboard", "data warehouse", "ETL", "OLAP"]
    }
    
    # Default examples for domains not in our mock database
    default_examples = ["metric", "indicator", "analysis", "benchmark", "trend"]
    
    return domain_examples.get(domain, default_examples)

# Define the "Act" node - Get jargon terms
def act(state: DomainDetectorState) -> DomainDetectorState:
    
    """Generate jargon terms for the current domain."""

    # Skip only if there is *no* domain to work on
    if not state["current_domain"]:
        new_state = state.copy()
        new_state["iteration"] += 1
        return new_state
    
    # Get example jargon terms
    example_terms = search_examples(state["current_domain"])
    
    act_template = """
    You are a domain expert in {domain}.
    
    Your task is to generate at least 5 jargon terms that are commonly used in the {domain} domain 
    but do NOT appear verbatim in the following column names: {columns}.
    
    Here are some example jargon terms for this domain: {examples}
    
    Return a JSON array of strings containing ONLY the jargon terms.
    Example: ["term1", "term2", "term3", "term4", "term5"]
    """
    
    act_prompt = PromptTemplate(
        template=act_template,
        input_variables=["domain", "columns", "examples"]
    )
    
    act_chain = LLMChain(llm=llm, prompt=act_prompt)
    
    jargon_response = act_chain.run({
        "domain": state["current_domain"],
        "columns": state["columns"],
        "examples": example_terms
    })
    
    # Extract the JSON array from the response
    try:
        # Find anything that looks like a JSON array
        match = re.search(r'\[.*\]', jargon_response, re.DOTALL)
        if match:
            jargon_terms = json.loads(match.group(0))
        else:
            jargon_terms = []
    except Exception:
        jargon_terms = []
    
    new_state = state.copy()
    new_state["jargon_terms"] = jargon_terms
    new_state["messages"].append(f"Generated jargon terms: {jargon_terms}")
    new_state["iteration"] += 1
    return new_state

# Define the "Reflect" node - Validate jargon terms
def reflect(state: DomainDetectorState) -> DomainDetectorState:
    """Validate jargon terms and update domains found."""
    
    if not state["current_domain"] or not state["jargon_terms"]:
        return state
    
    # Convert column names to lowercase for case-insensitive comparison
    columns_lower = [col.lower() for col in state["columns"]]
    
    # Filter out jargon terms that appear in column names
    valid_terms = []
    for term in state["jargon_terms"]:
        if term.lower() not in columns_lower:
            valid_terms.append(term)
    
    # If we have at least 3 valid terms, add the domain to our list
    new_state = state.copy()
    if len(valid_terms) >= 3:
        domain_entry = {
            "domain": state["current_domain"],
            "jargon_terms": valid_terms[:5]  # Limit to 5 terms
        }
        new_state["domains_found"].append(domain_entry)
        new_state["messages"].append(f"Added domain: {state['current_domain']} with terms: {valid_terms[:5]}")
        
        # Remove columns that were used for this domain (optional)
        # This is a simplified approach - in a real implementation, you might want
        # to use the LLM to determine which columns were used for this domain
        if len(new_state["remaining_columns"]) > 0:
            new_state["remaining_columns"].pop(0)
    else:
        new_state["messages"].append(f"Rejected domain: {state['current_domain']} - not enough valid jargon terms")
    
    # Reset current domain and jargon terms
    new_state["current_domain"] = None
    new_state["jargon_terms"] = []
    new_state["iteration"] += 1
    
    return new_state

# Define the condition to end the graph
def should_end(state: DomainDetectorState) -> str:
    """Determine if the graph should end."""
    if len(state["domains_found"]) >= 5:
        return "end"
    if not state["remaining_columns"]:
        return "end"
    if state["iteration"] >= 10:  # Safety limit
        return "end"
    return "continue"

# Build the graph
def build_domain_detector_graph():
    """Build and return the domain detector graph."""
    workflow = StateGraph(DomainDetectorState)
    
    # Add nodes
    workflow.add_node("think", think)
    workflow.add_node("act", act)
    workflow.add_node("reflect", reflect)
    
    # Add edges
    workflow.add_edge("think", "act")
    workflow.add_edge("act", "reflect")
    
    # Add conditional edge
    workflow.add_conditional_edges(
        "reflect",
        should_end,
        {
            "continue": "think",
            "end": END
        }
    )
    
    # Set the entry point
    workflow.set_entry_point("think")
    
    return workflow.compile()

# Function to run the domain detector
def run_domain_detector(columns, column_types=None):
    """
    Run the domain detector on the given columns.
    
    Args:
        columns (list): List of column names
        column_types (dict, optional): Dictionary mapping column names to their data types
    
    Returns:
        list: List of domains with their jargon terms
    """
    if column_types is None:
        column_types = {col: "text" for col in columns}
    
    # Initialize the state
    initial_state = DomainDetectorState(
        columns=columns,
        column_types=column_types
    )
    
    # Build and run the graph
    graph = build_domain_detector_graph()
    final_state = graph.invoke(initial_state)
    
    return final_state["domains_found"]

# Example usage
if __name__ == "__main__":
    columns = ["Quarter", "number_Customers", "Total_Transactions", "Revenue", "Profit"]
    column_types = {
        "Quarter": "object",
        "number_Customers": "int",
        "Total_Transactions": "float",
        "Revenue": "float",
        "Profit": "float"
    }
    
    domains = run_domain_detector(columns, column_types)
    print(json.dumps(domains, indent=2))