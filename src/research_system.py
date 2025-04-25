from langgraph.graph import StateGraph, END
from langsmith import traceable
import os
from dotenv import load_dotenv

load_dotenv()

def create_research_workflow():
    # avoid circular imports
    from tavily_search import run_research_agent
    from answer_drafter import run_draft_agent
    
    @traceable(name="Research Phase")
    def execute_research(state):
        try:
            result = run_research_agent(state["query"])
            research_data = result.get("output", "No results found")
            if not isinstance(research_data, str):
                research_data = str(research_data)
                
            return {
                "research_data": research_data,
                "query": state["query"] 
            }
        except Exception as e:
            print(f"Error in execute_research: {str(e)}")
            return {
                "research_data": f"Error in research: {str(e)}",
                "query": state["query"]
            }
    
    @traceable(name="Drafting Phase")
    def execute_draft(state):
        try:
            # print(f"State received in execute_draft: {state}") # Debug
            
            if "research_data" not in state or state["research_data"] is None:
                research_data = "No research data available"
            else:
                research_data = state["research_data"]
            
            query = state["query"]
            
            # calling the draft agent 
            final_answer = run_draft_agent(research_data, query)
            
            # Return the complete state
            return {
                "final_answer": final_answer,
                "research_data": research_data,
                "query": query
            }
        except Exception as e:
            print(f"Error in execute_draft: {str(e)}")
            return {
                "final_answer": f"Error in drafting: {str(e)}",
                "research_data": state.get("research_data", "No data"),
                "query": state.get("query", "No query")
            }
    
    builder = StateGraph(dict)
    builder.add_node("research", execute_research)
    builder.add_node("draft", execute_draft)
    builder.set_entry_point("research")
    builder.add_edge("research", "draft")
    builder.add_edge("draft", END)
    
    return builder.compile()

@traceable(name="Research Orchestrator")
def run_research_system(query):
    workflow = create_research_workflow()
    try:
        # exec the workflow 
        result = workflow.invoke({"query": query})
        
        if result is None:
            return {
                "research_data": "Workflow execution failed",
                "final_answer": "An error occurred during processing",
                "query": query
            }
        

        return {
            "research_data": result.get("research_data", "No research data"),
            "final_answer": result.get("final_answer", "No answer generated"),
            "query": query
        }
    except Exception as e:
        print(f"Error in research system: {str(e)}")
        return {
            "research_data": f"Error during research: {str(e)}",
            "final_answer": "An error occurred during the research process. Please try again.",
            "query": query
        }