# 민원 상담 AI API 서버

민원 상담 대화형 AI 모델을 위한 FastAPI 기반 API 서버입니다.

## 필요 파일 다운로드

AI Hub에서 다음 파일들을 다운로드 할 수 있습니다:
- [민간 민원 상담 LLM 사전학습 및 Instruction Tuning 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71844)

다운로드 후 다음 폴더들을 확인하세요:
- **1.모델소스코드**: 베이스 이미지 구성에 필요한 Dockerfile 및 소스코드
- **3.도커이미지**: 베이스 이미지를 로드하기 위한 파일

## 설치 및 실행 방법

### 1. 도커 이미지 구성

먼저 베이스 이미지를 로드합니다:
```bash
# 3.도커이미지 폴더에서
docker load -i [이미지 파일명].tar
```

### 2. API 서버 이미지 빌드

```bash
cd aihub_api_server
docker build -t consulting-api:latest .
```

### 3. 서버 실행

#### 일반 실행 (권장)
```bash
docker run -d --restart unless-stopped --gpus all -p 80:8000 consulting-api:latest python /nia_cun/src/api_server.py
```

#### ModuleNotFoundError 발생 시
필요한 패키지가 이미지에 설치되지 않은 경우 다음과 같이 실행합니다:

```bash
docker run -it --gpus all -p 80:8000 --entrypoint /bin/bash consulting-api:latest
cd /nia_cun/src

python -m pip install fastapi uvicorn pydantic
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install transformers accelerate peft
python -m pip install flash-attn

python api_server.py
```

설치 후에는 이 컨테이너를 커밋하여 새 이미지를 생성하는 것을 권장합니다:
```bash
# 다른 터미널에서 (컨테이너 ID 확인 후)
docker commit [컨테이너ID] consulting-api:latest
```

## API 사용 방법

### 엔드포인트

- `GET /`: 서버 상태 확인
- `GET /health`: 헬스체크
- `POST /api/predict`: 민원 상담 요청 처리

### 요청 예시

```bash
curl -X POST "http://<서버IP>:80/api/predict" \
     -H "Content-Type: application/json" \
     -d '{"question": "버스 노선 변경에 대해 문의합니다", "context": "30번 버스가 아파트 앞을 지나지 않아 불편합니다."}'
```

또는 브라우저에서 `http://<서버IP>:80/docs`에 접속하여 Swagger UI로 API를 테스트할 수 있습니다.

## 구성 파일 설명

- `api_server.py`: FastAPI 서버 구현
- `inference.py`: AI 모델 로딩 및 추론 로직
- `Dockerfile`: API 서버 이미지 빌드 파일

## 주의사항

- GPU가 필요합니다 (CUDA 지원)
- 충분한 디스크 공간이 필요합니다 (모델 크기에 따라 다름)
- 첫 실행 시 모델 로딩에 시간이 소요됩니다
