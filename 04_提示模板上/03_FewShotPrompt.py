"""欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 伏波"""

# 导入核心模块，使用最新的模块化路径
from langchain_core.prompts import (
    PromptTemplate,
    FewShotPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma  # 改进1：使用更简单的内存向量库 Chroma, Qdrant 需要额外安装和配置
from langchain_community.embeddings import OllamaEmbeddings # 改进2：为了清晰，单独导入
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector

# 1. 创建一些示例 (无变化)
samples = [
    {
        "flower_type": "玫瑰",
        "occasion": "爱情",
        "ad_copy": "玫瑰，浪漫的象征，是你向心爱的人表达爱意的最佳选择。"
    },
    {
        "flower_type": "康乃馨",
        "occasion": "母亲节",
        "ad_copy": "康乃馨代表着母爱的纯洁与伟大，是母亲节赠送给母亲的完美礼物。"
    },
    {
        "flower_type": "百合",
        "occasion": "庆祝",
        "ad_copy": "百合象征着纯洁与高雅，是你庆祝特殊时刻的理想选择。"
    },
    {
        "flower_type": "向日葵",
        "occasion": "鼓励",
        "ad_copy": "向日葵象征着坚韧和乐观，是你鼓励亲朋好友的最好方式。"
    }
]

# 2. 创建示例的格式化模板 (无变化)
example_prompt = PromptTemplate(
    input_variables=["flower_type", "occasion", "ad_copy"],
    template="鲜花类型: {flower_type}\n场合: {occasion}\n文案: {ad_copy}"
)

# 3. 初始化聊天模型和 Embedding 模型
#    注意：将模型分为两个，一个用于生成，一个用于嵌入
chat_model = ChatOllama(
    model="qwen3:8b", # 建议使用更新的 qwen2，或保持 qwen3:8b
    temperature=0
)

# 改进2：使用专门的、轻量级的 Embedding 模型
# 如果你没有 nomic-embed-text，请先在终端运行: ollama pull nomic-embed-text
# 中文场景下，m3e-base 也是不错的选择
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# 4. 创建基于语义相似度的示例选择器 (关键改进)
# 改进1：使用 Chroma.from_examples，它会自动在内存中创建向量数据库并填充示例
example_selector = SemanticSimilarityExampleSelector.from_examples(
    samples,          # 示例数据
    embeddings,       # 使用专用的 Embedding 模型
    Chroma,           # 使用 Chroma 作为向量存储
    k=1               # 选择最相似的1个示例
)

# 5. 创建最终的 FewShotPromptTemplate
# 改进4：优化 suffix，明确指示模型生成“文案”
final_prompt = FewShotPromptTemplate(
    example_selector=example_selector,  # 使用动态示例选择器
    example_prompt=example_prompt,
    prefix="根据下面的示例，为给定的鲜花类型和场合创作一句广告文案。", # 可以增加一个前缀指令
    suffix="鲜花类型: {flower_type}\n场合: {occasion}\n文案:", # 引导模型输出
    input_variables=["flower_type", "occasion"]
)

# 6. 创建并执行 LCEL 链
chain = final_prompt | chat_model | StrOutputParser()

print("--- 使用语义相似度选择示例 ---")
# 当输入与“爱情”相关时，应该会自动选择“玫瑰”作为示例
user_input = {"flower_type": "红玫瑰", "occasion": "情人节"}

# 我们可以先看看动态生成的完整 Prompt 是什么样子
print("--- 动态生成的完整 Prompt ---")
print(final_prompt.format(**user_input))
print("--------------------------\n")


# 执行链
result = chain.invoke(user_input)
print("--- 模型生成的结果 ---")
print(result)