from dotenv import load_dotenv
import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama 
# from tavily import TavilyClient # remove this line if intersted to use langchain_tavily and remove tavily client & @tool decorator and fn search  
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from typing import List
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
# agent = create_agent(model = llm, tools = tools)


#Pydantic 
""" Pydantic
-  base model is going to give us a baseclass so we can inherit from, in order to define a structured data schems
- It is going to provide functionality like data parsing and serialization and automatic type validations

- field class is going to allow us to add metadata to our models attributes
so, we can add descriptions, which is going to be helpful for llms to understand what to put in the field""" 

class Source(BaseModel):
    """ Schema for a source used by agent"""
    url:str= Field(description= "The URL of the source")


class AgentResponse(BaseModel):
    """ Schema for agent response with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory= list,
    description="List of sources used to generate answer"    )


agent = create_agent(model = llm, tools = tools, response_format=AgentResponse) 

def main():
    print("Hello from langchain-course!")
    response = agent.invoke({"messages": HumanMessage(content ="Search for 3 job postings for an AI engineer using langchain in the bay area on linkedin and list their details")})
    # response = agent.invoke({"messages": HumanMessage(content ="What is the weather in India?")})
    print(response)


if __name__ == "__main__":
    main()
