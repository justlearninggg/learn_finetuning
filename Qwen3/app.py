from fastapi import FastAPI
from pydantic import BaseModel
from model import QwenModel

app = FastAPI(title="Qwen3 Medical Reasoning API")

model = QwenModel()


class GenerateRequest(BaseModel):
    instruction: str
    input: str
    max_new_tokens: int = 512


@app.post("/generate")
def generate_text(req: GenerateRequest):
    response = model.generate(
        instruction=req.instruction,
        user_input=req.input,
        max_new_tokens=req.max_new_tokens
    )

    return {
        "response": response
    }
