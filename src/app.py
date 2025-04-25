import streamlit as st
import os
from research_system import run_research_system
from dotenv import load_dotenv

load_dotenv() 

# Set LangSmith tracing configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "NotPerplexity"

st.title("NotPerplexity - Agentic Web Search")
st.caption("Powered by LangChain, LangGraph, Tavily, and Gemini")

query = st.text_input("Enter your Search query:")

if st.button("Start Search"):
    if not query:
        st.warning("Please enter a search query")
    else:
        with st.spinner("Agentic Re-Search in progress..."):
            try:
                result = run_research_system(query)
                
                # Research data section with expandable details
                with st.expander("Research Results", expanded=False):
                    st.markdown(result["research_data"])
                
                # Final answer section
                st.subheader("Final Answer")
                st.markdown(result["final_answer"])
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please check your API keys and try again.")