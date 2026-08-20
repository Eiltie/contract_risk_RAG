import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# 显式指向项目根目录的 .env，避免依赖当前工作目录
load_dotenv()

def get_chat_model(temperature: float = 0.0):
    """拿「对话大模型」（DeepSeek），用于精排打分 + 生成结论。温度 0 = 输出稳定。"""
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=temperature,
    )

def get_embed_model():
    """拿「向量模型」（智谱），把一段文字变成一串数字（向量）。"""
    return OpenAIEmbeddings(
        model="embedding-3",
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
    )