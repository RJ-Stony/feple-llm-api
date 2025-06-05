# 로드된 기존 이미지를 베이스로 사용
FROM consulting_private:latest

# ENTRYPOINT 재정의
ENTRYPOINT []

# 작업 디렉토리 설정
WORKDIR /nia_cun/src

# API 서버 관련 패키지 설치
RUN pip install fastapi uvicorn pydantic

# API 서버 파일 복사
COPY api_server.py inference.py /nia_cun/src/

# 포트 설정
EXPOSE 8000

# API 서버 실행
CMD ["python", "/nia_cun/src/api_server.py"]