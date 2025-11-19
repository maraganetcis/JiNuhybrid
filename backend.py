from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from typing import List
import uvicorn

app = FastAPI(title="AI 챗봇 API", version="1.0.0")

# CORS 설정 (웹사이트에서 API 호출 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 모델
class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []

class ChatResponse(BaseModel):
    response: str
    model_used: str
    processing_time: float

# API 키 설정
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    AI 챗봇 API 엔드포인트
    """
    import time
    start_time = time.time()
    
    try:
        # 대화 기록을 컨텍스트로 포함
        context = ""
        if request.history:
            context = "이전 대화:\n"
            for msg in request.history[-5:]:  # 최근 5개만 사용
                role = "사용자" if msg["role"] == "user" else "어시스턴트"
                context += f"{role}: {msg['content']}\n"
            context += "\n"
        
        prompt = f"{context}현재 질문: {request.message}"
        
        # Gemini 모델 호출
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            response=response.text,
            model_used="gemini-1.5-flash",
            processing_time=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 처리 중 오류: {str(e)}")

@app.get("/")
async def root():
    return {"message": "AI 챗봇 API 서버가 실행 중입니다."}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
