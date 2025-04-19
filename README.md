#  Discourse Image Enhancement Service

简体中文 | [English](README.en.md)

为 [Discourse 图片增强插件](https://github.com/ShuiyuanSJTU/discourse-image-enhancement) 提供后端 AI 推理服务。

## 安装

1. 安装 Python 3.10 及以上版本。
2. [安装 pyTorch](https://pytorch.org/get-started/locally/)。
3. [安装 PaddlePaddle](https://github.com/PaddlePaddle/Paddle)。
4. 安装其他依赖：
    ```bash
    pip install -r requirements.txt
    ```

## 运行

* 直接运行（不推荐，仅用于调试）：
    ```bash
    python app.py
    ```
    初次运行后自动生成配置文件`config.json`，请根据实际情况修改配置。

* 使用 uvicorn
    ```bash
    uvicorn app:app --port 8000
    ```
    由于 paddleOCR 已知有[内存泄漏问题](https://github.com/PaddlePaddle/PaddleOCR/issues/11639)，不建议采用该方法部署到生产环境。

* 使用 gunicorn
    ```bash
    gunicorn -w 2 -k uvicorn.workers.UvicornWorker --max-requests 1000 --timeout 120 --bind 0.0.0.0:8000 app:app
    ```

## API

访问`/docs`以查看 API 文档。

### GET `/health`

检查服务是否正常运行。如果服务正常运行，返回`{"status": "ok"}`。

### POST `/analyze`

分析图片的 Endpoint，需要将此 Endpoint 完整 URL 填入[Discourse 图片增强插件](https://github.com/ShuiyuanSJTU/discourse-image-enhancement)的`image enhancement analyze service endpoint`设置。

请求参数 (`application/json`)：
```
{
  "images": [
    {
      "sha1": "string",
      "url": "string",
      "id": 0
    }
  ],
  "ocr": true,
  "embedding": true,
  "lang": "string"
}
```

响应结果 (`application/json`)：
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
      "embedding": [
        0
      ]
    }
  ],
  "message": ""
}
```

## License
Apache-2.0

* [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Apache-2.0 license
* [OnnxOCR](https://github.com/jingsongliujing/OnnxOCR) - Apache-2.0 license
* [Bert](https://github.com/google-research/bert) - Apache-2.0 license
* [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) - MIT license
