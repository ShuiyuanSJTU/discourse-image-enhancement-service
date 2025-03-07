from fastapi import FastAPI
from backend.model import RequestBody, ResponseBody
from backend.analyzer import ImageAnalyzer
from backend.config import init_config
from threading import Lock
import uvicorn

config = init_config()

app = FastAPI()
inference_lock = Lock()

analyzer = ImageAnalyzer(config=config)

@app.post("/analyze")
@app.post("/analyze/")
def analyze(request_body: RequestBody) -> ResponseBody:
    with inference_lock:
        results = analyzer.analyze_images(
            request_body.images, analyze_ocr=request_body.ocr,
            analyze_description=request_body.description
        )
        return ResponseBody(images=results)

@app.get("/health")
@app.get("/health/")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)