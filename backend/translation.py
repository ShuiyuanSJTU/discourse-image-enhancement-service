from transformers import pipeline
from typing import List
import requests
import re

class BaseTranslation:
    def infer(self, text: str):
        raise NotImplementedError
    
    def batch_infer(self, text: List[str]):
        raise NotImplementedError

class TranslationLocalPipeline(BaseTranslation):
    def __init__(self, model: str = "Helsinki-NLP/opus-mt-en-zh"):
        self.model = pipeline("translation", model=model)

    def infer(self, text: str):
        return self.model(text)[0]['translation_text']
    
    def batch_infer(self, text: List[str]):
        return [result['translation_text'] for result in self.model(text)]
    
class TranslationLLMAgent(BaseTranslation):
    DEFAULT_PROMPT = '你是一个翻译机器人。你是一个图片搜索框架的一部分，用于对模型进行预处理，结果将被存储在数据库。我们的AI应用会根据用户上传的图像生成中文的文字描述图片中的内容，在你之前，已有模型将图片描述成英文，你需要将用户的英文输入翻译成中文。你的输出结果将被保存在数据库中，稍后用户会输入图片关键词，我们将从数据库中检索你的输出结果，以向用户展示对应图片。考虑到用户可能使用不同的同义词描述一张图片，你可以多次翻译同一段描述，以增加多样性，不同尝试之间以回车分隔，最多输出5种不同翻译，不要对结果进行编号。请注意，你只直接输出文本的翻译结果，不要尝试与用户进行对话，不要输出无关内容。'

    @staticmethod
    def remove_line_numbers(text: str) -> str:
        lines = text.split('\n')
        processed_lines = []
        for line in lines:
            processed_line = re.sub(r'^\d+\.\s*', '', line)
            processed_lines.append(processed_line)
        return '\n'.join(processed_lines)

    def __init__(self, model: str, endpoint: str, 
                 api_key: str, 
                 temperature: float = 0.1,
                 max_tokens: int = 1024,
                 prompt: str = DEFAULT_PROMPT):
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt = prompt.strip() if prompt.strip() else self.DEFAULT_PROMPT

    def infer(self, text: str):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

        data = {
            'messages': [
                {'role': 'system', 'content': self.prompt},
                {'role': 'user', 'content': text}
            ],
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'model': self.model
        }

        response = requests.post(self.endpoint, headers=headers, json=data)

        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
        else:
            raise RuntimeError(f"Error: {response.status_code} {response.text}")

        return self.remove_line_numbers(content)

    def batch_infer(self, text: List[str]):
        return [self.infer(t) for t in text]
