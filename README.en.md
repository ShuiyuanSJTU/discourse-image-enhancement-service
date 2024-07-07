# Discourse Image Enhancement Service

[简体中文](README.md) | English

Provides backend AI inference services for the [Discourse Image Enhancement Plugin](https://github.com/ShuiyuanSJTU/discourse-image-enhancement).

## Installation

1. Install Python 3.10 or above.
2. [Install pyTorch](https://pytorch.org/get-started/locally/).
3. [Install PaddlePaddle](https://github.com/PaddlePaddle/Paddle).
4. Install other dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running

* Direct run (not recommended, for debugging only):
    ```bash
    python app.py
    ```
    The configuration file `config.json` will be automatically generated after the first run. Modify the configuration as needed.

* Using uvicorn:
    ```bash
    uvicorn app:app --port 8000
    ```
    Due to a known [memory leak issue](https://github.com/PaddlePaddle/PaddleOCR/issues/11639) in paddleOCR, it is not recommended to use this method for production deployment.

* Using gunicorn:
    ```bash
    gunicorn -w 2 -k uvicorn.workers.UvicornWorker --max-requests 1000 --timeout 120 --bind 0.0.0.0:8000 app:app
    ```

## API

Visit `/docs` to view the API documentation.

### GET `/health`

Check if the service is running normally. If the service is running, it returns `{"status": "ok"}`.

### POST `/analyze`

Endpoint for image analysis. The complete URL of this endpoint needs to be filled in the `image enhancement analyze service endpoint` setting of the [Discourse Image Enhancement Plugin](https://github.com/ShuiyuanSJTU/discourse-image-enhancement).

Request parameters (`application/json`):
```
{
  "images": [
    {
      "sha1": "string",
      "url": "string",
      "id": 0
    }
  ],
  "lang": "string"
}
```

Response result (`application/json`):
```
{
  "images": [
    {
      "sha1": "string",
      "success": true,
      "error": "string",
      "ocr_result": [
        "string"
      ],
      "description": "string"
    }
  ],
  "message": ""
}
```

---