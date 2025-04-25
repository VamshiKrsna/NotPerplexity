# answer_drafter.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv() 

def create_drafting_workflow():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.4
    )
    
    # state graph from langgraph
    builder = StateGraph(dict)
    
    def analyze_research(state):
        try:
            # print(f"analyze_research received state: {state}") # for debug
            
            research_data = state.get("research_data", "No research data available")
            query = state.get("query", "No query provided")
            
            messages = [
                SystemMessage(content="You are a research analyst. Analyze this data:"),
                HumanMessage(content=f"Research Data: {research_data}\nQuery: {query}")
            ]
            response = llm.invoke(messages)
            analysis = response.content
            
            return {"analysis": analysis}

        except Exception as e:
            print(f"Error in analyze_research: {str(e)}")
            return {"analysis": f"Error analyzing research: {str(e)}"}

    def draft_answer(state):
        try:
            # print(f"draft_answer received state: {state}") # Debug 
            
            research_data = state.get("research_data", "No research data available")
            analysis = state.get("analysis", "No analysis available")
            query = state.get("query", "No query provided")
            
            messages = [
                SystemMessage(content="""You are a technical writer. Create a comprehensive answer 
                based on the provided analysis and research data. Format your response with 
                Markdown for readability."""),
                HumanMessage(content=f"""
                Research Data: {research_data}
                
                Analysis: {analysis}
                
                Query: {query}
                """)
            ]
            response = llm.invoke(messages)
            final = response.content
            
            return {"final_answer": final}

        except Exception as e:
            print(f"Error in draft_answer: {str(e)}")
            return {"final_answer": f"Error drafting answer: {str(e)}"}

    builder.add_node("analyze", analyze_research)
    builder.add_node("draft", draft_answer)
    builder.add_edge("analyze", "draft")
    builder.set_entry_point("analyze")
    builder.set_finish_point("draft")
    
    return builder.compile()

def run_draft_agent(research_data, query):
    try:
        # print(f"Running draft agent with research_data: {research_data[:100]}... and query: {query}")  # Debug print to see inputs
        
        workflow = create_drafting_workflow()
        
        if not isinstance(research_data, str):
            research_data = str(research_data)
        
        initial_state = {
            "research_data": research_data,
            "query": query,
            "analysis": ""  
        }
        
        # exec workflow
        result = workflow.invoke(initial_state)
        
        # print(f"Draft agent workflow result: {result}") # debug 
        
        # Check if result is None
        if result is None:
            return "Error: Workflow returned None"
            
        # Return the final answer
        return result.get("final_answer", "No answer generated")
    except Exception as e:
        print(f"Error in draft agent: {str(e)}")
        return f"Error generating answer: {str(e)}"