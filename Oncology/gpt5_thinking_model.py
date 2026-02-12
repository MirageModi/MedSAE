"""
OpenAI GPT-5 Model Integration for Medical LLM Generation

This module implements OpenAI's GPT-5 using the Responses API.

Important:
- Only include `text.format` with `"type": "json_schema"` when a *non-null* schema
  object is provided. Passing `schema: null` causes a 400:
  "Invalid type for 'text.format.schema': expected an object, but got null instead."
"""

from openai import OpenAI
import time
from typing import Dict, Optional, Any
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GPT5Config:
    """Configuration for OpenAI GPT-5 model."""
    model_name: str = "gpt-5-2025-08-07"  # Actual model name based on logs
    max_output_tokens: int = 8192  # Increased to allow reasoning + response
    reasoning_effort: str = "medium"  # "low" | "medium" | "high"
    text_format: str = "text"  # "text" | "json_object" | "json_schema"
    text_verbosity: str = "medium"  # "low" | "medium" | "high"

class GPT5Model:
    """
    GPT-5 wrapper for medical tasks using OpenAI Responses API.
    """
    
    def __init__(self, config: GPT5Config):
        self.config = config
        self.client = OpenAI()
        
    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a response with GPT-5 using the Responses API.
        Uses structured messages and the reasoning parameter.
        """
        try:
            start_time = time.time()

            text_arg: Dict[str, Any] = {"verbosity": self.config.text_verbosity}
            if schema is not None:
                text_arg["format"] = {
                    "type": "json_schema",
                    "name": "ClinicalDecision",
                    "schema": schema,
                    "strict": True,
                }

            resp = self.client.responses.create(
                model=self.config.model_name,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_output_tokens=self.config.max_output_tokens,
                reasoning={"effort": self.config.reasoning_effort},
                text=text_arg,
            )
            generation_time = time.time() - start_time
            logger.info(f"Response object:\n{resp}")
            text = ""
            reasoning_content = ""
            
            logger.info(f"Response object attributes: {dir(resp)}")
            
            if hasattr(resp, "output_text") and resp.output_text:
                text = resp.output_text
                logger.info(f"Extracted via output_text (first 100 chars): {text[:100]}")
            else:
                logger.info(f"output_text not available or empty. Has attribute: {hasattr(resp, 'output_text')}, Value: {getattr(resp, 'output_text', None)}")
                
                logger.info(f"Response has output: {hasattr(resp, 'output')}")
                
                if hasattr(resp, "output") and resp.output:
                    logger.info(f"Output array length: {len(resp.output)}")
                    
                    for idx, output_item in enumerate(resp.output):
                        logger.info(f"Output item {idx} type: {getattr(output_item, 'type', 'no type')}")
                        
                        item_type = getattr(output_item, "type", None)
                        
                        if item_type == "message":
                            content_array = getattr(output_item, "content", [])
                            logger.info(f"Message content array length: {len(content_array) if content_array else 0}")
                            
                            if content_array:
                                for content_part in content_array:
                                    content_type = getattr(content_part, "type", None)
                                    logger.info(f"Content part type: {content_type}")
                                    
                                    if content_type in ("text", "output_text"):
                                        part_text = getattr(content_part, "text", "")
                                        if part_text:
                                            text += part_text
                                            logger.info(f"Extracted text from message content (first 100 chars): {part_text[:100]}")
                        
                        elif item_type == "reasoning":
                            content_array = getattr(output_item, "content", [])
                            logger.info(f"Reasoning content array length: {len(content_array) if content_array else 0}")
                            
                            if content_array:
                                for content_part in content_array:
                                    content_type = getattr(content_part, "type", None)
                                    logger.info(f"Reasoning content part type: {content_type}")
                                    
                                    if content_type in ("text", "output_text"):
                                        part_text = getattr(content_part, "text", "")
                                        if part_text:
                                            reasoning_content += part_text
                                            logger.info(f"Extracted reasoning text (first 100 chars): {part_text[:100]}")
                
                if resp.status == "incomplete":
                    logger.warning(f"Response incomplete: {resp.incomplete_details}")
                    logger.warning(f"Usage: {resp.usage}")
                    
                    if resp.incomplete_details and resp.incomplete_details.reason == "max_output_tokens":
                        logger.error(f"Insufficient max_output_tokens: {resp.max_output_tokens}")
                        logger.error(f"Try increasing max_output_tokens. Current: {resp.max_output_tokens}, Used: {resp.usage.output_tokens}")
                        
                        text = f"[Response incomplete due to max_output_tokens limit. Need higher token limit for complete response. Used: {resp.usage.output_tokens}/{resp.max_output_tokens}]"
                
                elif not text:
                    logger.warning(f"No text extracted from response")
                    logger.warning(f"Response status: {getattr(resp, 'status', 'unknown')}")
                    logger.warning(f"Response output items: {[getattr(item, 'type', 'unknown') for item in resp.output] if hasattr(resp, 'output') and resp.output else 'none'}")
                    
                    try:
                        full_dump = resp.model_dump() if hasattr(resp, 'model_dump') else str(resp)
                        logger.warning(f"Full response dict: {full_dump}")
                        
                        if hasattr(resp, 'reasoning') and resp.reasoning:
                            logger.info(f"Found reasoning field: {type(resp.reasoning)}")
                            logger.info(f"Reasoning attributes: {dir(resp.reasoning) if hasattr(resp.reasoning, '__dir__') else 'N/A'}")
                    except Exception as e:
                        logger.warning(f"Could not dump response: {e}")
                
            return {
                "response": text,
                "reasoning": reasoning_content if reasoning_content else None,
                "generation_time": generation_time,
                "model_config": {
                    "max_output_tokens": self.config.max_output_tokens,
                    "reasoning_effort": self.config.reasoning_effort,
                    "text_format": self.config.text_format,
                    "text_verbosity": self.config.text_verbosity,
                },
            }
        except Exception as e:
            logger.error(f"Error generating response with GPT-5: {e}", exc_info=True)
            return {
                "response": "",
                "error": str(e),
                "generation_time": 0,
            }


def create_gpt5_config(
    reasoning_effort: str = "medium",
    text_format: str = "text",
    text_verbosity: str = "medium"
) -> GPT5Config:
    return GPT5Config(
        reasoning_effort=reasoning_effort,
        text_format=text_format,
        text_verbosity=text_verbosity
    )
