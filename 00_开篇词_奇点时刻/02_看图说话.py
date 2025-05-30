"""
本项目使用开源多模态模型 llava:13b，通过本地 Ollama 部署，支持图像输入与文本输出任务。
"""

import requests
from PIL import Image
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from langchain_ollama import ChatOllama
from transformers import BlipProcessor, BlipForConditionalGeneration

# 指定要使用的工具模型（HuggingFace的image-caption模型）
hf_model = "Salesforce/blip-image-captioning-large"

# 初始化处理器和工具模型
# 预处理器将准备图像供模型使用
processor = BlipProcessor.from_pretrained(hf_model)
# 然后我们初始化工具模型本身
model = BlipForConditionalGeneration.from_pretrained(hf_model)


class ImageCapTool(BaseTool):
    name: str = "Image captioner"
    description: str = "为图片创作说明文案."

    def _run(self, url: str) -> str:
        # 下载图像并将其转换为PIL对象
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        # 预处理图像
        inputs = processor(image, return_tensors="pt")
        # 生成字幕
        out = model.generate(**inputs, max_new_tokens=20)
        # 获取字幕
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


llm = ChatOllama(
    base_url="http://localhost:11434",  # 服务地址
    model="qwen3:8b",  # 模型名称
    temperature=0
)

# 使用工具初始化智能体并运行
tools = [ImageCapTool()]
agent = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
)
img_url = 'https://mir-s3-cdn-cf.behance.net/project_modules/hd/eec79e20058499.563190744f903.jpg'
agent.invoke(input=f"{img_url}\n请创作合适的中文推广文案")
