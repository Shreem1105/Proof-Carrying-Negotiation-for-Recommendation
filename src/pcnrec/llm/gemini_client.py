import os
import time
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pcnrec.utils.logging import setup_logger

logger = setup_logger(__name__)

class GeminiClient:
    def __init__(self, config):
        """
        Initialize Gemini Client from google-genai SDK.
        """
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set in environment.")
        
        self.client = genai.Client(api_key=api_key)
        
        self.model_name = config['llm']['model']
        self.temperature = config['llm']['temperature']
        self.max_tokens = config['llm']['max_output_tokens']
        self.timeout = config['llm']['timeout_s']
        self.max_retries = config['llm']['max_retries']

    @property
    def _retry_decorator(self):
        return retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(Exception), # Broad retry for transient API issues, refine if needed
            reraise=True
        )

    def generate_text(self, prompt: str, system_instruction: str = None) -> str:
        """
        Generates free-text response.
        """
        @self._retry_decorator
        def _call_api():
            config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                system_instruction=system_instruction,
                response_mime_type="text/plain"
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            return response.text

        try:
            return _call_api()
        except Exception as e:
            logger.error(f"Gemini generate_text failed after retries: {e}")
            raise

    def generate_structured(self, prompt: str, schema_model, system_instruction: str = None):
        """
        Generates structured output parsed into schema_model (Pydantic).
        """
        @self._retry_decorator
        def _call_api():
            # Generate and sanitize schema manually to remove additionalProperties
            try:
                schema = schema_model.model_json_schema()
            except AttributeError:
                # Fallback for Pydantic V1 or other types
                schema = schema_model.schema()

            def sanitize(s):
                if isinstance(s, dict):
                    s.pop('additionalProperties', None)
                    for k, v in s.items():
                        if isinstance(v, (dict, list)):
                            sanitize(v)
                        if k == 'items' and isinstance(v, dict):
                            sanitize(v)
            
            sanitize(schema)

            config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=schema
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            # The SDK handles parsing if response_schema is provided as Pydantic model class
            # But wait, google-genai usually returns a parsed object if configured correctly 
            # OR we parse text.
            # In latest google-genai, response.parsed is available if schema is passed.
            
            if hasattr(response, 'parsed') and response.parsed is not None:
                if isinstance(response.parsed, dict):
                    return schema_model(**response.parsed)
                return response.parsed
            else:
                # Fallback: parse text manually if SDK didn't auto-parse to instance
                import json
                try:
                    data = json.loads(response.text)
                    return schema_model(**data)
                except Exception as parse_error:
                    logger.error(f"Failed to parse JSON: {response.text[:100]}...")
                    raise parse_error

        try:
            return _call_api()
        except Exception as e:
            logger.error(f"Gemini generate_structured failed after retries: {e}")
            raise
