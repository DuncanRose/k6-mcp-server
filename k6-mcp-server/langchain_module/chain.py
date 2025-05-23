import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.chat_models import ChatAnthropic, ChatOllama

def get_llm(model_name: str, provider: str):
    if provider == "openai":
        return ChatOpenAI(model=model_name)
    elif provider == "azure":
        return AzureChatOpenAI(
            deployment_name=model_name,
            openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version="2023-05-15",
            openai_api_key=os.getenv("AZURE_OPENAI_KEY")
        )
    elif provider == "anthropic":
        return ChatAnthropic(model=model_name)
    elif provider == "ollama":
        return ChatOllama(model=model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def build_chain(model_name: str = "gpt-4", provider: str = "openai") -> Runnable:
    prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer: {question}")
    llm = get_llm(model_name, provider)
    chain = prompt | llm | StrOutputParser()
    return chain
