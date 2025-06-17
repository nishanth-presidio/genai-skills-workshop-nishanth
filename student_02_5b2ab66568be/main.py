from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Alaska FAQ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Alaska FAQ RAG API"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    # Temporary mock response
    return QuestionResponse(
        answer=f"Mock response for: {request.question}. RAG system will be connected."
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
