import unittest
from unittest.mock import patch, MagicMock
from backend.description import DescriptionPipeline
from PIL import Image

class TestDescriptionPipeline(unittest.TestCase):

    def setUp(self):
        self.test_image = Image.new('RGB', (60, 30), color = 'red')
    
    @patch('backend.description.pipeline')
    def test_infer(self, mock_pipeline):
        mock_pipeline.return_value = lambda x: [{'generated_text': 'This is a test description.'}]
        dp = DescriptionPipeline()
        result = dp.infer(self.test_image)
        self.assertEqual(result, 'This is a test description.')
        mock_pipeline.assert_called_once_with("image-to-text", model="Salesforce/blip-image-captioning-large")

    @patch('backend.description.pipeline')
    def test_batch_infer(self, mock_pipeline):
        mock_pipeline.return_value = lambda x: [[{'generated_text': 'This is a test description.'}], [{'generated_text': 'This is a test description.'}]]
        dp = DescriptionPipeline()
        result = dp.batch_infer([self.test_image, self.test_image])
        self.assertEqual(result, ['This is a test description.', 'This is a test description.'])
        mock_pipeline.assert_called_once_with("image-to-text", model="Salesforce/blip-image-captioning-base")

if __name__ == '__main__':
    unittest.main()