from langchain_ollama import ChatOllama

client = ChatOllama(
    base_url="http://localhost:11434",  # 服务地址
    model="qwen3:8b",  # 模型名称
    temperature=0.5,
)

response = client.invoke("请给我的花店起个名")
print(response.content)
