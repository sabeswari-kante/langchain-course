from dotenv import load_dotenv
import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama 
# from tavily import TavilyClient # remove this line if intersted to use langchain_tavily and remove tavily client & @tool decorator and fn search  
from langchain_tavily import TavilySearch

load_dotenv() 

# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args: 
#         query: the query to seach for
#     Return:
#         The search result
#     """
#     print(f"searching for {query}")
#     return tavily.search(query=query) # "India weather is sunny"

# creating agent from here
# llm = ChatOpenAI()
# llm = ChatOllama(model="llama3")
llm = ChatOllama(model="qwen2.5")
tools = [TavilySearch()] # [TavilySearch()] [search]
agent = create_agent(model = llm, tools = tools)

def main():
    print("Hello from langchain-course!")
    response = agent.invoke({"messages": HumanMessage(content ="Search for 3 job postings for an AI engineer using langchain in the bay area on linkedin and list their details")})
    # response = agent.invoke({"messages": HumanMessage(content ="What is the weather in India?")})
    print(response)


if __name__ == "__main__":
    main()
