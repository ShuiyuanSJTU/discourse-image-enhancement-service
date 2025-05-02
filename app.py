from fastapi import FastAPI
from backend.model import ImageAnalysisRequestBody, ImageAnalysisResponseBody, TextEmbeddingRequestBody, TextEmbeddingResponseBody, ImageEmbeddingResponseBody, ImageEmbeddingRequestBody
from backend.analyzer import ImageAnalyzer
from backend.config import init_config
from backend.utils import extract_url_encoded_image
from threading import Lock
import uvicorn

config = init_config()

app = FastAPI()
inference_lock = Lock()

analyzer = ImageAnalyzer(config=config)

@app.post("/analyze")
@app.post("/analyze/")
def analyze(request_body: ImageAnalysisRequestBody) -> ImageAnalysisResponseBody:
    with inference_lock:
        results = analyzer.analyze_images(
            request_body.images, analyze_ocr=request_body.ocr,
            analyze_embedding=request_body.embedding
        )
        return ImageAnalysisResponseBody(images=results)

@app.post("/text_embedding")
@app.post("/text_embedding/")
def text_embedding(request_body: TextEmbeddingRequestBody) -> TextEmbeddingResponseBody:
    with inference_lock:
        embedding = analyzer.clip_processor.infer(request_body.text)
        return TextEmbeddingResponseBody(embedding=embedding.tolist())

@app.post("/image_embedding")
@app.post("/image_embedding/")
def image_embedding(request_body: ImageEmbeddingRequestBody) -> ImageEmbeddingResponseBody:
    image = extract_url_encoded_image(request_body.image)
    with inference_lock:
        embedding = analyzer.clip_processor.infer(image)
        return ImageEmbeddingResponseBody(embedding=embedding.tolist())

@app.get("/health")
@app.get("/health/")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)