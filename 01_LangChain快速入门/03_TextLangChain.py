from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.8
)
# predict 已经废弃，建议使用 invoke
response = llm.predict("请给我的花店起个名")
print(response)
