"""
使用Ollama的ChatOllama模型生成文本
"""

from langchain_ollama import ChatOllama

llm = ChatOllama(
    base_url="http://localhost:11434",  # 服务地址
    model="qwen3:8b"  # 模型名称
)
# 同步调用模型生成文本
text = llm.invoke("请给我写一句情人节红玫瑰的中文宣传语")
print(text)