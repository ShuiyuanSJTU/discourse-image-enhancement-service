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
    description: Optional[str] = None

@dataclass
class ProcessingImage:
    sha1: str
    info: ImageInfo
    result: AnalysisResult
    image: Optional[Image] = None

class RequestBody(BaseModel):
    images: List[ImageInfo]
    lang: Optional[str] = None

class ResponseBody(BaseModel):
    images: List[AnalysisResult]
    message: str = ""