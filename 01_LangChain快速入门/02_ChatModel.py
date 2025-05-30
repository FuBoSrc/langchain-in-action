"""
指定 system message
并尝试使用 HumanMessage
"""
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(
    base_url='localhost:11434',  # 服务地址
    model='qwen3:8b',  # 模型名称
    temperature=0.8
)
response = llm.invoke(
    [
        SystemMessage(content="You are a creative AI."),
        HumanMessage(content="请给我的花店起个名")
    ]
)
print(response.content)
