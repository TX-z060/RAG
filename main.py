from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

app = FastAPI()

class AskRequest(BaseModel):
    question:str


@app.get("/")
def read_root():
    return {"message":"课程资料问答助手已启动"}

@app.get("/ask")
def ask(question:str):
    return {"question":question}


@app.post("/ask")
def ask_post(data:AskRequest):
    return {"question":data.question}

@app.get("/materials")
def read_material():
    file_path = Path(__file__).with_name("course.txt")
    content = file_path.read_text(encoding="utf-8")
    return {"content":content}

@app.get("/search")
def search_material(keyword:str):
    file_path = Path(__file__).with_name("course.txt")
    content = file_path.read_text(encoding="utf-8")

    keyword = keyword.strip()
    matches = []

    for line in content.splitlines():
        if keyword and keyword.lower() in line.lower():
            matches.append(line)
    
    return {"keyword":keyword,"matches":matches}