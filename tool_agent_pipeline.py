# tool_agent_pipeline.py
import json
from typing import Callable, Dict, Any

# ==========================================
# 1. DEFINE TOOLS (Functions available to LLM)
# ==========================================
def search_news_database(query: str, category: str = "Sci/Tech") -> str:
    """Searches the local news archive for articles matching a query and category."""
    print(f"\n[TOOL EXECUTION] Running search_news_database with query='{query}', category='{category}'")
    
    # Mock database results
    mock_db = {
        "Sci/Tech": "Found 3 articles: 1. Apple unveils new AI chip. 2. Quantum computing milestone reached. 3. SpaceX launches orbital satellite.",
        "Sports": "Found 2 articles: 1. Champions League finals scheduled. 2. World cup qualifiers update.",
        "Business": "Found 1 article: 1. Stock markets hit record highs."
    }
    return mock_db.get(category, "No matching articles found.")

# Registry of available tools
TOOL_REGISTRY: Dict[str, Callable] = {
    "search_news_database": search_news_database
}

# ==========================================
# 2. SIMULATE LLM DECISION & TOOL CALLING
# ==========================================
def llm_agent_router(user_request: str) -> Dict[str, Any]:
    """
    Simulates the LLM receiving a user request and deciding whether 
    to call a tool or answer directly.
    """
    print(f"--- Step 1: User Request Received ---\nUser: {user_request}\n")
    print("--- Step 2: LLM Processing & Tool Decision ---")
    
    # In a full framework (like LangChain, LlamaIndex, or OpenAI Function Calling), 
    # the LLM outputs a structured JSON tool call if a tool is needed.
    # Here we demonstrate the decision logic based on the user prompt:
    
    if "search" in user_request.lower() or "find" in user_request.lower():
        # LLM decides to call the 'search_news_database' tool
        tool_decision = {
            "action": "tool_call",
            "tool_name": "search_news_database",
            "arguments": {
                "query": "AI chip or technology",
                "category": "Sci/Tech"
            }
        }
        print(f"LLM Decision: Invoke tool '{tool_decision['tool_name']}' with args {tool_decision['arguments']}")
        return tool_decision
    else:
        # LLM decides to answer directly without tools
        return {
            "action": "direct_response",
            "output": "I can help you analyze or search news articles. Please let me know what you'd like to find!"
        }

# ==========================================
# 3. COMPLETE AGENTIC EXECUTION LOOP
# ==========================================
def run_agent_workflow(user_request: str):
    # Step 1 & 2: User Request -> LLM -> Tool Decision
    decision = llm_agent_router(user_request)
    
    if decision["action"] == "tool_call":
        tool_name = decision["tool_name"]
        args = decision["arguments"]
        
        # Step 3: Tool Execution (Tool Result)
        if tool_name in TOOL_REGISTRY:
            tool_function = TOOL_REGISTRY[tool_name]
            tool_result = tool_function(**args)
            
            print(f"\n--- Step 3: Tool Result Obtained ---")
            print(f"Result:\n{tool_result}")
            
            # Step 4: Feed Tool Result back to LLM for Final Response
            print(f"\n--- Step 4: LLM Synthesizing Final Response ---")
            final_response = (
                f"Based on your request ('{user_request}'), I searched our database "
                f"and retrieved the following information:\n\n{tool_result}\n\n"
                "Let me know if you would like me to classify or summarize any of these articles!"
            )
            print(f"Final Response:\n{final_response}")
            return final_response
        else:
            raise ValueError(f"Tool '{tool_name}' not found in registry.")
    else:
        print(f"\n--- Step 4: Final Response (Direct) ---")
        print(decision["output"])
        return decision["output"]

if __name__ == "__main__":
    # Test the pipeline with a request that triggers tool usage
    sample_request = "Can you search our database for recent technology news articles?"
    run_agent_workflow(sample_request)