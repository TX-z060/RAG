from pathlib import Path
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
import os
load_dotenv(Path(__file__).with_name(".env"))


model = ChatDeepSeek(
    model="deepseek-flash", 
    timeout = 60
    )

response =model.invoke("请只回复：连接成功")
print(response.content)