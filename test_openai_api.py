# Usage:
# pytest test_openai_api.py -v

import pytest
import os
import openai
import json
import re
from typing import Dict, List, Any, Optional

# Mock response for testing without making actual API calls
MOCK_COMPLETION_RESPONSE = {
    "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
    "object": "text_completion",
    "created": 1589478378,
    "model": "gpt-3.5-turbo",
    "choices": [
        {
            "message": {
                "content": "This is a test response from the OpenAI API.",
                "role": "assistant"
            },
            "finish_reason": "stop",
            "index": 0
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 10,
        "total_tokens": 15
    }
}

# Configure with your API key for actual API calls
# openai.api_key = os.environ.get("OPENAI_API_KEY")

class TestOpenAIResponse:
    """Test suite for validating OpenAI API responses"""
    
    @pytest.fixture
    def mock_response(self):
        """Provide a mock response for testing"""
        return MOCK_COMPLETION_RESPONSE
    
    def test_response_structure(self, mock_response):
        """Test that the response has the expected structure"""
        assert "id" in mock_response
        assert "object" in mock_response
        assert "created" in mock_response
        assert "choices" in mock_response
        assert isinstance(mock_response["choices"], list)
        assert len(mock_response["choices"]) > 0
        assert "usage" in mock_response
    
    def test_message_content_exists(self, mock_response):
        """Test that the message content exists and is a string"""
        content = mock_response["choices"][0]["message"]["content"]
        assert content is not None
        assert isinstance(content, str)
        assert len(content) > 0
    
    def test_token_usage_within_bounds(self, mock_response):
        """Test that token usage is within expected bounds"""
        usage = mock_response["usage"]
        assert 0 <= usage["prompt_tokens"] <= 1000  # Adjust bounds as needed
        assert 0 <= usage["completion_tokens"] <= 2000  # Adjust bounds as needed
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    
    def test_finish_reason_valid(self, mock_response):
        """Test that finish reason is one of the expected values"""
        finish_reason = mock_response["choices"][0]["finish_reason"]
        valid_reasons = ["stop", "length", "content_filter", "function_call", "null"]
        assert finish_reason in valid_reasons
    
    def test_content_meets_requirements(self, mock_response):
        """Test that content meets specific requirements
        Adjust these tests based on your specific content requirements"""
        content = mock_response["choices"][0]["message"]["content"]
        
        # Example constraints - modify based on your needs
        assert len(content) >= 5  # Min length
        assert len(content) <= 1000  # Max length
        assert not re.search(r'(badword1|badword2)', content, re.IGNORECASE)  # No forbidden words
    
    @pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0])
    def test_live_api_temperature(self, temperature):
        """Test actual API calls with different temperature settings
        Note: This test requires a valid API key and will make actual API calls"""
        pytest.skip("Skip live API tests by default - remove this line to enable")
        
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello"}],
            temperature=temperature,
        )
        
        # Convert response object to dict for testing
        response_dict = json.loads(response.model_dump_json())
        
        # Basic validation
        assert response_dict["choices"][0]["message"]["content"]
        
        # Keep track of responses to check if temperature affects output
        return response_dict["choices"][0]["message"]["content"]
    
    def test_response_time(self):
        """Test that the API responds within an acceptable time range"""
        pytest.skip("Skip live API tests by default - remove this line to enable")
        
        import time
        client = openai.OpenAI()
        
        start_time = time.time()
        client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Quick test"}],
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 10  # Response should be under 10 seconds
    
    def test_custom_parameters(self, mock_response, custom_param=None):
        """Template for testing custom parameters"""
        # Example of how to test a custom parameter
        if custom_param:
            # Add custom validation logic here
            pass
        
        # Default pass for template function
        assert True

# Helper function for actual API calls when needed
def make_actual_api_call(prompt: str, **kwargs) -> Dict[str, Any]:
    """Make an actual call to the OpenAI API
    
    Args:
        prompt: The prompt to send to the API
        **kwargs: Additional parameters to pass to the API
        
    Returns:
        API response as a dictionary
    """
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=kwargs.get("model", "gpt-3.5-turbo"),
        messages=[{"role": "user", "content": prompt}],
        temperature=kwargs.get("temperature", 0.7),
        max_tokens=kwargs.get("max_tokens", 150),
    )
    
    # Convert response object to dict
    return json.loads(response.model_dump_json())

if __name__ == "__main__":
    pytest.main(["-v"])