from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from langchain_ollama import ChatOllama

chat = ChatOllama(
    model="qwen3:8b",  # 模型名称
    temperature=0.8
)
messages = [
    SystemMessage(content="你是一个很棒的智能助手"),
    HumanMessage(content="请给我的花店起个名")
]
response = chat(messages)
print(response)
