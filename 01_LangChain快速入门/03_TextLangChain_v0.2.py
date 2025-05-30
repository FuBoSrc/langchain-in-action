from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3:8b",  # 模型名称
    temperature=0.8,
    # token 设置这么点结果瞬间就不准确咯，但 qwen3:8b 模型的最大 token 数量为 4096
    num_predict=60
)
response = llm.predict("请给我的花店起个名")
print(response)
