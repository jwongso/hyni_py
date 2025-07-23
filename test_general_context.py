"""
Functional and integration tests for GeneralContext using pytest.

These tests verify the functionality of the GeneralContext class with actual API calls
and mock responses. They test various features including multi-turn conversations,
streaming, multimodal support, and provider-specific functionality.
"""

import pytest
import json
import os
import time
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sseclient
from io import BytesIO

# Import the GeneralContext class (assuming it's in a module named general_context)
from general_context import GeneralContext, ContextConfig, SchemaException, ValidationException


@dataclass
class StreamingResponse:
    """Structure to hold streaming response data."""
    events: List[str] = field(default_factory=list)
    complete_content: str = ""
    finished: bool = False
    error: bool = False
    error_message: str = ""


def get_api_key_for_provider(provider: str) -> str:
    """
    Get API key for a specific provider from environment variables.

    Args:
        provider: Provider name (e.g., 'claude', 'openai', 'deepseek')

    Returns:
        API key string or empty string if not found
    """
    env_mapping = {
        'claude': 'CL_API_KEY',
        'openai': 'OPENAI_API_KEY',
        'deepseek': 'DEEPSEEK_API_KEY'
    }

    env_var = env_mapping.get(provider, f"{provider.upper()}_API_KEY")
    api_key = os.environ.get(env_var, '')

    # Try loading from .hynirc if not in environment
    if not api_key:
        rc_path = Path.home() / '.hynirc'
        if rc_path.exists():
            try:
                with open(rc_path, 'r') as f:
                    for line in f:
                        if line.strip().startswith(env_var):
                            api_key = line.split('=', 1)[1].strip()
                            break
            except Exception:
                pass

    return api_key


def make_api_call(url: str, api_key: str, payload: Dict[str, Any],
                  is_anthropic: bool = False, timeout: int = 60) -> str:
    """
    Make a simple API call for testing purposes.

    Args:
        url: API endpoint URL
        api_key: API key for authentication
        payload: Request payload as dictionary
        is_anthropic: Whether this is an Anthropic API call
        timeout: Request timeout in seconds

    Returns:
        Response text

    Raises:
        Exception: If the API call fails
    """
    headers = {
        'Content-Type': 'application/json'
    }

    if is_anthropic:
        headers['anthropic-version'] = '2023-06-01'
        headers['x-api-key'] = api_key
    else:
        headers['Authorization'] = f'Bearer {api_key}'

    # Set up retry strategy
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    try:
        response = session.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout
        )
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        raise Exception(f"API call failed: {str(e)}")


def make_streaming_api_call(url: str, api_key: str, payload: Dict[str, Any],
                           streaming_response: Optional[StreamingResponse] = None,
                           is_anthropic: bool = False, timeout: int = 60) -> str:
    """
    Make an API call with streaming support.

    Args:
        url: API endpoint URL
        api_key: API key for authentication
        payload: Request payload as dictionary
        streaming_response: Optional StreamingResponse object to populate
        is_anthropic: Whether this is an Anthropic API call
        timeout: Request timeout in seconds

    Returns:
        Response text (empty if streaming)

    Raises:
        Exception: If the API call fails
    """
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
    }

    if is_anthropic:
        headers['anthropic-version'] = '2023-06-01'
        headers['x-api-key'] = api_key
    else:
        headers['Authorization'] = f'Bearer {api_key}'

    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout,
            stream=streaming_response is not None
        )
        response.raise_for_status()

        if streaming_response:
            # Handle SSE streaming
            # For streaming responses, we need to iterate over the response lines
            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode('utf-8')

                # SSE format: "data: {json_content}"
                if line_str.startswith('data: '):
                    data = line_str[6:]  # Remove "data: " prefix

                    if data == '[DONE]':
                        streaming_response.finished = True
                        break

                    try:
                        event_json = json.loads(data)
                        streaming_response.events.append(data)

                        # Extract content from delta
                        if 'delta' in event_json and 'text' in event_json['delta']:
                            streaming_response.complete_content += event_json['delta']['text']

                        # Check for errors
                        if 'error' in event_json:
                            streaming_response.error = True
                            streaming_response.error_message = event_json['error'].get('message', 'Unknown error')

                    except json.JSONDecodeError:
                        pass

            return ""
        else:
            return response.text

    except requests.exceptions.RequestException as e:
        raise Exception(f"API call failed: {str(e)}")


def create_test_image() -> bytes:
    """
    Create a small test image (1x1 pixel PNG).

    Returns:
        PNG image data as bytes
    """
    # 1x1 pixel red PNG
    png_data = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
        0x54, 0x08, 0x99, 0x01, 0x01, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01,
        0xE2, 0x21, 0xBC, 0x33, 0x00, 0x00, 0x00, 0x00,
        0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82
    ])
    return png_data


class TestGeneralContextFunctional:
    """Functional tests for GeneralContext class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment before each test."""
        # Get API key
        api_key = os.environ.get('CL_API_KEY')
        if not api_key:
            rc_path = Path.home() / '.hynirc'
            if rc_path.exists():
                with open(rc_path, 'r') as f:
                    for line in f:
                        if line.strip().startswith('CL_API_KEY'):
                            api_key = line.split('=', 1)[1].strip()
                            break

        if not api_key:
            pytest.skip("CL_API_KEY environment variable not set")

        self.api_key = api_key
        self.test_schema_dir = "schemas/"

        # Create context with validation enabled
        config = ContextConfig(
            enable_validation=True,
            default_max_tokens=100,
            default_temperature=0.3
        )

        schema_path = os.path.join(self.test_schema_dir, "claude.json")
        self.context = GeneralContext(schema_path, config)
        self.context.set_api_key(api_key)

        # Clean up any test files
        test_image_path = Path("test_image.png")
        if test_image_path.exists():
            test_image_path.unlink()

    def test_schema_loading_and_context_creation(self):
        """Test basic schema loading and context creation."""
        # Test context creation
        assert self.context is not None
        assert self.context.supports_multimodal()
        assert self.context.supports_system_messages()
        assert self.context.supports_streaming()

        # Test provider info using getter methods
        assert self.context.get_provider_name() == "claude"
        assert self.context.get_endpoint() != ""

    def test_pre_loaded_schema_constructor(self):
        """Test creating context with pre-loaded schema."""
        # Load schema manually
        schema_path = os.path.join(self.test_schema_dir, "claude.json")
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        # Create context with pre-loaded schema
        config = ContextConfig(enable_validation=True)
        context = GeneralContext(schema, config)

        assert context.get_provider_name() == "claude"
        assert context.supports_multimodal()

    def test_basic_single_message(self):
        """Test basic single message conversation."""
        # Set up a simple conversation
        self.context.add_user_message("Hello, please respond with exactly 'Hi there!'")

        # Validate request structure
        assert self.context.is_valid_request()

        request = self.context.build_request()

        # Verify request structure
        assert "model" in request
        assert "max_tokens" in request
        assert "messages" in request
        assert len(request["messages"]) == 1
        assert request["messages"][0]["role"] == "user"

        # Perform actual API call
        api_url = self.context.get_endpoint()
        is_anthropic = self.context.get_provider_name() == "claude"

        try:
            response_str = make_api_call(api_url, self.api_key, request, is_anthropic)
            response_json = json.loads(response_str)
        except Exception as ex:
            pytest.fail(f"API call failed: {ex}")

        # Extract and validate response
        text = self.context.extract_text_response(response_json)
        assert text
        assert text.strip() == "Hi there!"

    def test_multi_turn_conversation(self):
        """Test multi-turn conversation."""
        # First exchange
        self.context.add_user_message("What's 2+2?")

        request1 = self.context.build_request()
        assert len(request1["messages"]) == 1

        # Simulate assistant response
        self.context.add_assistant_message("2+2 equals 4.")

        # Second user message
        self.context.add_user_message("What about 3+3?")

        request2 = self.context.build_request()
        assert len(request2["messages"]) == 3

        # Verify message order and roles
        assert request2["messages"][0]["role"] == "user"
        assert request2["messages"][1]["role"] == "assistant"
        assert request2["messages"][2]["role"] == "user"

        assert self.context.is_valid_request()

    def test_system_message(self):
        """Test system message functionality."""
        system_prompt = "You are a helpful assistant that responds concisely."
        self.context.set_system_message(system_prompt)
        self.context.add_user_message("Hello")

        request = self.context.build_request()

        # Check how system message is handled
        if "system" in request:
            # Anthropic Claude style - separate system field
            assert request["system"] == system_prompt
            assert len(request["messages"]) == 1
            assert request["messages"][0]["role"] == "user"
        else:
            # OpenAI style - system message as first message
            assert len(request["messages"]) == 2
            assert request["messages"][0]["role"] == "system"
            assert request["messages"][0]["content"] == system_prompt
            assert request["messages"][1]["role"] == "user"

        assert self.context.is_valid_request()

    def test_parameter_handling(self):
        """Test parameter setting and validation."""
        # Test valid parameters
        self.context.set_parameter("temperature", 0.7)
        self.context.set_parameter("max_tokens", 150)
        self.context.set_parameter("top_p", 0.9)

        self.context.add_user_message("Test message")

        request = self.context.build_request()

        assert request["temperature"] == 0.7
        assert request["max_tokens"] == 150
        assert request["top_p"] == 0.9

        # Test parameter getter methods
        assert self.context.get_parameter("temperature") == 0.7
        assert self.context.get_parameter_as("max_tokens", int) == 150
        assert self.context.has_parameter("top_p")

        # Test parameter with default
        assert self.context.get_parameter_as("non_existent", str, "default") == "default"

        # Test null parameter validation
        with pytest.raises(ValidationException) as exc_info:
            self.context.set_parameter("temperature", None)
        assert "cannot be null" in str(exc_info.value)

        # Test parameter validation based on schema
        # These would need to be defined in the schema to work
        # with pytest.raises(ValidationException):
        #     self.context.set_parameter("temperature", 2.0)

    def test_model_selection(self):
        """Test model selection."""
        # Test valid model
        self.context.set_model("claude-3-5-haiku-20241022")
        self.context.add_user_message("Hello")

        request = self.context.build_request()
        assert request["model"] == "claude-3-5-haiku-20241022"

        # Test invalid model
        with pytest.raises(ValidationException):
            self.context.set_model("invalid-model")

        # Test supported models list
        models = self.context.get_supported_models()
        assert models
        assert "claude-3-5-sonnet-20241022" in models

    def test_multimodal_image_handling(self):
        """Test image handling (multimodal)."""
        # Create test image
        png_data = create_test_image()
        with open("test_image.png", "wb") as f:
            f.write(png_data)

        # Test with image file path
        self.context.add_user_message("What do you see in this image?",
                                     "image/png", "test_image.png")

        request = self.context.build_request()
        assert len(request["messages"]) == 1

        content = request["messages"][0]["content"]
        assert len(content) == 2  # text + image

        # Verify text content
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "What do you see in this image?"

        # Verify image content
        assert content[1]["type"] == "image"
        assert content[1]["source"]["media_type"] == "image/png"
        assert "data" in content[1]["source"]

        # Clean up
        Path("test_image.png").unlink()

    def test_validation_errors(self):
        """Test validation errors."""
        # Test empty context (no messages)
        errors = self.context.get_validation_errors()
        assert errors
        assert not self.context.is_valid_request()

        # Add message and test again
        self.context.add_user_message("Hello")
        errors = self.context.get_validation_errors()
        assert not errors
        assert self.context.is_valid_request()

    def test_context_reset(self):
        """Test context reset functionality."""
        # Set up context with data
        self.context.set_system_message("Test system")
        self.context.set_parameter("temperature", 0.8)
        self.context.add_user_message("Hello")
        self.context.add_assistant_message("Hi")

        # Verify data is present
        request_before = self.context.build_request()
        assert len(request_before["messages"]) == 2
        assert request_before["temperature"] == 0.8
        assert "system" in request_before

        # Reset context
        self.context.reset()

        # Verify data is cleared
        errors = self.context.get_validation_errors()
        assert errors  # Should have validation errors due to no messages

        request_after = self.context.build_request()
        assert len(request_after["messages"]) == 0
        assert request_after.get("temperature", 0) != 0.8

    def test_clear_methods(self):
        """Test individual clear methods."""
        # Set up context with data
        self.context.set_system_message("Test system")
        self.context.set_parameter("temperature", 0.8)
        self.context.add_user_message("Hello")

        # Test clear_user_messages
        self.context.clear_user_messages()
        assert len(self.context.get_messages()) == 0
        assert self.context.get_parameter("temperature") == 0.8  # Parameters should remain

        # Re-add message and test clear_system_message
        self.context.add_user_message("Hello")
        self.context.clear_system_message()
        request = self.context.build_request()
        assert "system" not in request
        assert len(request["messages"]) == 1

        # Test clear_parameters
        self.context.clear_parameters()
        assert not self.context.has_parameter("temperature")

    def test_response_parsing(self):
        """Test response parsing with mock responses."""
        # Create mock successful response
        mock_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello! How can I help you?"}],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 15, "output_tokens": 8}
        }

        # Test text extraction
        text = self.context.extract_text_response(mock_response)
        assert text == "Hello! How can I help you?"

        # Test full response extraction
        content = self.context.extract_full_response(mock_response)
        assert isinstance(content, list)
        assert len(content) == 1

        # Test error response
        error_response = {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Missing required field: max_tokens"
            }
        }

        error_msg = self.context.extract_error(error_response)
        assert error_msg == "Missing required field: max_tokens"

    def test_edge_cases_and_errors(self):
        """Test edge cases and error conditions."""
        # Test very long message
        long_message = "a" * 10000
        self.context.add_user_message(long_message)
        assert self.context.is_valid_request()

        # Test special characters
        self.context.clear_user_messages()
        self.context.add_user_message("Hello 世界! 🌍 Special chars: @#$%^&*()")
        assert self.context.is_valid_request()

        # Test empty message
        self.context.clear_user_messages()
        self.context.add_user_message("")
        assert self.context.is_valid_request()

        # Test clearing individual components
        self.context.add_user_message("Test")
        self.context.set_parameter("temperature", 0.5)

        self.context.clear_user_messages()
        request = self.context.build_request()
        assert len(request["messages"]) == 0
        assert request.get("temperature") == 0.5  # Parameters should remain

        self.context.clear_parameters()
        request = self.context.build_request()
        assert request.get("temperature", 0) != 0.5

    def test_getter_methods(self):
        """Test all getter methods."""
        # Test schema getter
        schema = self.context.get_schema()
        assert isinstance(schema, dict)
        assert "provider" in schema

        # Test headers getter
        headers = self.context.get_headers()
        assert isinstance(headers, dict)
        assert "Anthropic-Version" in headers

        # Test parameters getter
        self.context.set_parameter("test_param", "test_value")
        params = self.context.get_parameters()
        assert params["test_param"] == "test_value"

        # Test messages getter
        self.context.add_user_message("Test")
        messages = self.context.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_api_key_handling(self):
        """Test API key handling."""
        # Test has_api_key
        assert self.context.has_api_key()

        # Test empty API key validation
        with pytest.raises(ValidationException) as exc_info:
            self.context.set_api_key("")
        assert "API key cannot be empty" in str(exc_info.value)

        # Test API key in headers
        self.context.set_api_key("test_key_123")
        headers = self.context.get_headers()
        assert "test_key_123" in headers.get("x-api-key", "")

    def test_base64_validation(self):
        """Test base64 validation logic."""
        # Create a proper base64 string
        test_data = b"Hello, World!"
        base64_str = base64.b64encode(test_data).decode('utf-8')

        # Test with data URI
        data_uri = f"data:image/png;base64,{base64_str}"
        assert self.context._is_base64_encoded(data_uri)

        # Test with plain base64
        assert self.context._is_base64_encoded(base64_str)

        # Test with invalid strings
        assert not self.context._is_base64_encoded("not base64!")
        assert not self.context._is_base64_encoded("")
        assert not self.context._is_base64_encoded("short")

    def test_performance(self):
        """Performance test for request building."""
        import time

        start = time.time()

        # Build many requests
        for i in range(1000):
            if i % 100 == 0:
                self.context.clear_user_messages()
            self.context.add_user_message(f"Message {i}")
            request = self.context.build_request()

        end = time.time()
        duration = (end - start) * 1000  # Convert to milliseconds

        # Should be able to build requests quickly
        assert duration < 1000  # Less than 1 second for 1000 requests

    def test_actual_api_integration(self):
        """Integration test with actual API."""
        self.context.set_system_message("Respond with exactly 'Integration test successful'")
        self.context.add_user_message("Please confirm this integration test is working.")

        request = self.context.build_request()

        api_url = self.context.get_endpoint()
        is_anthropic = self.context.get_provider_name() == "claude"
        assert is_anthropic

        try:
            response_str = make_api_call(api_url, self.api_key, request, is_anthropic)
            response_json = json.loads(response_str)
        except Exception as ex:
            pytest.fail(f"API call failed: {ex}")

        # Extract and validate response
        text = self.context.extract_text_response(response_json)
        assert text
        assert text.strip() == "Integration test successful"

    def test_real_multi_turn_conversation(self):
        """Test real multi-turn conversation with actual API."""
        api_url = self.context.get_endpoint()
        is_anthropic = self.context.get_provider_name() == "claude"

        # First turn
        self.context.add_user_message("What's the capital of France?")
        request1 = self.context.build_request()

        try:
            response_str1 = make_api_call(api_url, self.api_key, request1, is_anthropic)
            response_json1 = json.loads(response_str1)
        except Exception as ex:
            pytest.fail(f"First API call failed: {ex}")

        # Extract and add assistant response
        text1 = self.context.extract_text_response(response_json1)
        assert text1
        self.context.add_assistant_message(text1)

        # Second turn
        self.context.add_user_message("What's the population of that city?")
        request2 = self.context.build_request()

        try:
            response_str2 = make_api_call(api_url, self.api_key, request2, is_anthropic)
            response_json2 = json.loads(response_str2)
        except Exception as ex:
            pytest.fail(f"Second API call failed: {ex}")

        # Extract second response
        text2 = self.context.extract_text_response(response_json2)
        assert text2

        # Verify the response mentions Paris and population
        text2_lower = text2.lower()
        assert any(word in text2_lower for word in ["paris", "million", "population"])

    @pytest.mark.skipif(not Path("tests/german.png").exists(),
                        reason="Test image file not found")
    def test_real_image_handling(self):
        """Test real image handling with Claude."""
        # Skip if not using Claude
        if self.context.get_provider_name() != "claude":
            pytest.skip("Skipping image test for non-Claude provider")

        api_url = self.context.get_endpoint()
        is_anthropic = True

        self.context.add_user_message("Describe this image in exactly 5 words.",
                                     "image/png", "tests/german.png")
        request = self.context.build_request()

        try:
            response_str = make_api_call(api_url, self.api_key, request, is_anthropic)
            response_json = json.loads(response_str)
        except Exception as ex:
            pytest.fail(f"API call with image failed: {ex}")

        # Extract response
        text = self.context.extract_text_response(response_json)
        assert text

        # Count words
        word_count = len(text.split())

        # The model might not follow instructions exactly
        assert word_count <= 10

    def test_streaming_parameter(self):
        """Test streaming parameter and functionality."""
        if not self.context.supports_streaming():
            pytest.skip("Provider doesn't support streaming")

        api_url = self.context.get_endpoint()
        is_anthropic = self.context.get_provider_name() == "claude"

        # Test 1: Verify streaming parameter is set correctly
        self.context.set_parameter("stream", True)
        self.context.add_user_message("Count from 1 to 5, explaining each number.")

        request = self.context.build_request()
        assert "stream" in request
        assert request["stream"] is True

        # Test 2: Actually test streaming functionality
        streaming_response = StreamingResponse()

        try:
            make_streaming_api_call(api_url, self.api_key, request,
                                   streaming_response, is_anthropic)

            # Verify streaming worked
            assert len(streaming_response.events) > 0, "Should receive multiple streaming events"
            assert streaming_response.complete_content, "Should have accumulated content"
            assert not streaming_response.error, f"Should not have errors: {streaming_response.error_message}"

            # Verify we received incremental updates
            assert len(streaming_response.events) > 1, "Should receive multiple chunks for streaming"

            print(f"Received {len(streaming_response.events)} streaming events")
            print(f"Complete content length: {len(streaming_response.complete_content)}")

        except Exception as e:
            pytest.fail(f"Streaming test failed: {e}")

        # Reset for other tests
        self.context.set_parameter("stream", False)

    def test_complex_prompt_handling(self):
        """Test with very complex prompts."""
        api_url = self.context.get_endpoint()
        is_anthropic = self.context.get_provider_name() == "claude"

        # Create a complex prompt with special characters, code blocks, etc.
        complex_prompt = """
        # Test Document

        This is a *complex* prompt with **markdown** and `code`.

        ```python
        def hello_world():
            print("Hello, world!")
            return 42
        ```

        Special characters: €£¥$@#%^&*()_+{}|:"<>?~`-=[]\\;',./

        Please respond with:
        1. The number of lines in the Python function
        2. The exact string that would be printed
        """

        self.context.add_user_message(complex_prompt)
        request = self.context.build_request()

        try:
            response_str = make_api_call(api_url, self.api_key, request, is_anthropic)
            response_json = json.loads(response_str)
        except Exception as ex:
            pytest.fail(f"API call with complex prompt failed: {ex}")

        # Extract response
        text = self.context.extract_text_response(response_json)
        assert text

        # Verify response contains expected information
        assert "2" in text  # Number of lines
        assert "Hello, world!" in text  # Printed string


def test_multi_provider_support():
    """Test support for multiple providers."""
    providers = ["mistral", "openai", "deepseek"]

    for provider in providers:
        api_key = get_api_key_for_provider(provider)
        if not api_key:
            print(f"Skipping {provider} test: No API key available")
            continue

        try:
            # Create context for this provider
            config = ContextConfig(
                enable_validation=True,
                default_max_tokens=50
            )

            schema_path = f"schemas/{provider}.json"
            if not Path(schema_path).exists():
                print(f"Skipping {provider}: Schema file not found")
                continue

            context = GeneralContext(schema_path, config)
            context.set_api_key(api_key)
            assert context is not None

            # Set up a simple request
            context.add_user_message("Respond with exactly one word: 'Success'")

            request = context.build_request()
            api_url = context.get_endpoint()
            is_anthropic = (provider == "claude")

            # Make API call
            response_str = make_api_call(api_url, api_key, request, is_anthropic)
            response_json = json.loads(response_str)

            # Extract response
            text = context.extract_text_response(response_json)
            assert text
            assert "Success" in text

        except Exception as ex:
            pytest.fail(f"Provider {provider} test failed: {ex}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
