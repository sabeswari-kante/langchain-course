from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain_openai import ChatOpenAI 
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
In LangChain, chains act as step-by-step workflows that complete tasks in sequence. Each step can use an LLM to process or transform data and then interact with external tools if needed. Built-in chains are like ready-made templates for common tasks. They link together LLMs, prompts and tools using pre-defined logic, helping developers save time by avoiding manual setup for every step.

Common Built-in Chains:

LLMChain: The most basic chain. It simply takes an input, formats it using a prompt and passes it to an LLM to get a response.
Sequential Chains: These link multiple sub-chains together, where the output of one step automatically becomes the input for the next. It's great for breaking down a complex problem.
"""
    summary_template = """
    Given the information {information} abput a person I want you to create :
    1. Give a short summary
    """

    summary_prompt_template = PromptTemplate(
        input_variables=['information'],
        template= summary_template
    )

    # llm= ChatOpenAI(temperature = 0, model="gpt-5-mini") #gpt-5 
    llm= ChatOllama(temperature =0, model = "gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke(input = {"information": information})
    print(response.content)


if __name__ == "__main__":
    main()

