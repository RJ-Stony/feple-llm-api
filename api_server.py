from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
import uvicorn
import time
import os

# 기존 inference.py에서 필요한 기능 임포트
from inference import inference

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="민원 상담 AI API", 
    description="민원 상담 관련 질의에 응답하는 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (프로덕션에서는 특정 도메인으로 제한)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    context: Optional[str] = None

@app.get("/")
async def root():
    return {
        "message": "민원 상담 AI API 서비스가 실행 중입니다",
        "usage": "POST /api/predict 엔드포인트를 사용해 질문을 전송하세요",
        "status": "healthy"
    }

@app.post("/api/predict")
async def predict(query: QueryRequest):
    start_time = time.time()
    try:
        # 질문과 컨텍스트 합치기
        question = query.question
        if query.context:
            question = f"{question}\ncontext: {query.context}"
            
        logger.info(f"질문 받음: {question[:100]}...")
        
        # inference 함수 호출
        response = inference(question)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"응답 생성 완료: {response[:100]}... (처리 시간: {processing_time:.2f}초)")
        return {"response": response, "processing_time": f"{processing_time:.2f}초"}
        
    except Exception as e:
        logger.error(f"에러 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

# 요청 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"요청 받음: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    end_time = time.time()
    logger.info(f"응답 전송: {request.method} {request.url.path} - 상태 코드: {response.status_code}, 처리 시간: {end_time - start_time:.2f}초")
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"API 서버 시작 중... 포트: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)