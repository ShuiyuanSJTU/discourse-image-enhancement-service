from pydantic import BaseModel
from dataclasses import dataclass
from typing import List, Optional
from PIL.Image import Image

class ImageInfo(BaseModel):
    sha1: str
    url: str
    id: Optional[int] = None

class AnalysisResult(BaseModel):
    sha1: str
    success: bool = True
    error: Optional[str] = None
    ocr_result: Optional[List[str]] = None
    embedding: Optional[List[float]] = None

@dataclass
class ProcessingImage:
    sha1: str
    info: ImageInfo
    result: AnalysisResult
    image: Optional[Image] = None

class ImageAnalysisRequestBody(BaseModel):
    images: List[ImageInfo]
    ocr: bool = True
    embedding: bool = True
    lang: Optional[str] = None

class ImageAnalysisResponseBody(BaseModel):
    images: List[AnalysisResult]
    message: str = ""

class TextEmbeddingRequestBody(BaseModel):
    text: str

class ImageEmbeddingRequestBody(BaseModel):
    image: str

class TextEmbeddingResponseBody(BaseModel):
    embedding: List[float]
    message: str = ""
    success: bool = True

class ImageEmbeddingResponseBody(BaseModel):
    embedding: List[float]
    message: str = ""
    success: bool = True