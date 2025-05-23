from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from langchain_module.chain import build_chain

app = FastAPI()
router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    model: str = "gpt-4"
    provider: str = "openai"

@router.post("/ask")
async def ask_llm(query: QueryRequest):
    chain = build_chain(model_name=query.model, provider=query.provider)
    answer = chain.invoke({"question": query.question})
    return {"provider": query.provider, "model": query.model, "answer": answer}

app.include_router(router)
