import unittest
from unittest.mock import patch, MagicMock
from backend.generation.llm_provider import OllamaClient

class TestOllamaClient(unittest.TestCase):
    
    @patch('backend.generation.llm_provider.requests.get')
    def test_check_connection_success(self, mock_get):
        mock_get.return_value.status_code = 200
        client = OllamaClient()
        self.assertTrue(client.check_connection())
        
    @patch('backend.generation.llm_provider.requests.get')
    def test_check_connection_failure(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        client = OllamaClient()
        self.assertFalse(client.check_connection())

    @patch('backend.generation.llm_provider.requests.post')
    def test_generate_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "The sky is blue because..."}
        mock_post.return_value = mock_response
        
        client = OllamaClient()
        response = client.generate("Why?")
        self.assertEqual(response, "The sky is blue because...")
        
    @patch('backend.generation.llm_provider.requests.post')
    def test_generate_error(self, mock_post):
        mock_post.side_effect = Exception("API Error")
        client = OllamaClient()
        response = client.generate("Why?")
        self.assertTrue(response.startswith("Error:"))

if __name__ == '__main__':
    unittest.main()
