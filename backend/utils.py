from base64 import b64decode
from PIL import Image
from io import BytesIO

def extract_url_encoded_image(url: str) -> Image:
    """
    Extracts a URL-encoded image from a data URL.
    """
    _, data = url.split(",", 1)
    image_content = b64decode(data)
    return Image.open(BytesIO(image_content)).convert('RGB')