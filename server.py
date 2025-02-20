from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.prompt
    response = generate_response(user_input)
    save_to_memory(user_input, response)
    return {"response": response}
