# from langchain.agents import AgentExecutor 
# from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages 
# from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_google_genai import ChatGoogleGenerativeAI
# import os
# from dotenv import load_dotenv

# load_dotenv() 
 
# def create_reserach_agent():
#     tools = [
#         TavilySearchResults(
#             api_key = os.getenv("TAVILY_API_KEY"),
#             max_results = 3, 
#             name = "crawler"
#         )
#     ]
#     llm = ChatGoogleGenerativeAI(
#         model = "gemini-1.5-flash",
#         google_api_key = os.getenv("GEMINI_API_KEY"),
#         temperature = 0,
#     )# we need accurate, non hallucinating answers 
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "user", """You are a senior research analyst. 
#                 Gather comprehensive information about: {input}.
#                 Use multiple search queries if needed.
#                 Validate sources and prioritize most recent data.
#                 Answer in third person perspective like a news reporter.
#                 """
#             )
#         ]
#     )
#     agent = (
#         {
#             "input":lambda x:x["input"],
#             "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
#         }
#         | prompt
#         | llm.bind_tools(tools = tools)
#         | OpenAIToolsAgentOutputParser()
#     )
#     return AgentExecutor(agent = agent, tools = tools)

# def run_research_agent(query):
#     executor = create_reserach_agent() 
#     return executor.invoke(
#         {
#             "input":query
#         }
#     )



# tavily_search.py
from langchain.agents import AgentExecutor 
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages 
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv() 

# Create a singleton to avoid recreating the agent unnecessarily
_research_agent = None

def create_research_agent():
    global _research_agent
    if _research_agent is not None:
        return _research_agent
        
    # Define the search tool without extra parameters that might cause schema warnings
    search_tool = TavilySearchResults(
        api_key=os.getenv("TAVILY_API_KEY"),
        max_results=3
    )
    
    tools = [search_tool]
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", 
                """You are a senior research analyst. Your job is to gather comprehensive information about 
                the user's query. Use search tools to find accurate and recent information. Present your 
                findings in a clear, organized format that a news reporter would use."""
            ),
            (
                "user", "{input}"
            ),
            (
                "placeholder", "{agent_scratchpad}"
            )
        ]
    )
    
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
        }
        | prompt
        | llm.bind_tools(tools=tools)
        | OpenAIToolsAgentOutputParser()
    )
    
    _research_agent = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return _research_agent

def run_research_agent(query):
    executor = create_research_agent()
    try:
        result = executor.invoke({"input": query})

        if "output" not in result or not result["output"]:
            result["output"] = "No research results found"
        return result
    except Exception as e:
        print(f"Error in research agent: {str(e)}")
        return {"output": f"Error during research: {str(e)}"}