"""
Gemini API Integration for AutoGen-based Talent Matching Tool

This module provides integration with Google's Gemini API for AutoGen agents.
"""

import os
import json
import google.generativeai as genai
from typing import List, Dict, Any, Tuple, Optional, Union
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gemini_integration")

class GeminiLLMConfig:
    """Configuration for Gemini LLM integration with AutoGen."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        """
        Initialize the Gemini LLM configuration.
        
        Args:
            api_key: Gemini API key
            model_name: Gemini model name to use
        """
        self.api_key = api_key
        self.model_name = model_name
        self._initialize_gemini()
        logger.info(f"Initialized Gemini API with model: {model_name}")
    
    def _initialize_gemini(self):
        """Initialize the Gemini API client."""
        genai.configure(api_key=self.api_key)
        
        # Set default generation config
        self.generation_config = {
            "temperature": 0.2,  # Low temperature for more deterministic responses
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
    
    def get_gemini_for_autogen(self) -> Dict[str, Any]:
        """
        Get the Gemini configuration for AutoGen.
        
        Returns:
            Dictionary with configuration for AutoGen
        """
        return {
            "config_list": [{
                "model": self.model_name,
                "api_key": self.api_key,
                "api_type": "gemini"
            }],
            "temperature": 0.2,
            "cache_seed": 42
        }
    
    def create_gemini_completion_function(self):
        """
        Create a completion function for AutoGen to use with Gemini.
        
        Returns:
            A function that can be used as a completion_function for AutoGen
        """
        def completion_function(messages: List[Dict[str, str]], model: str, **kwargs) -> Dict[str, Any]:
            """
            Custom completion function for AutoGen to use Gemini.
            
            Args:
                messages: List of message dictionaries for the conversation
                model: Model name (will be ignored in favor of self.model_name)
                **kwargs: Additional parameters for the generation
            
            Returns:
                Dictionary with the completion result
            """
            try:
                # Initialize model
                model = genai.GenerativeModel(self.model_name)
                
                # Combine messages into a prompt
                prompt = self._format_messages_for_gemini(messages)
                
                # Generate response
                response = model.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                    **kwargs
                )
                
                return {
                    "content": response.text,
                    "role": "assistant",
                    "function_call": None,
                }
            except Exception as e:
                logger.error(f"Error in Gemini completion: {str(e)}")
                return {
                    "content": "I encountered an error processing your request. Please try again.",
                    "role": "assistant",
                    "function_call": None,
                }
        
        return completion_function
    
    def _format_messages_for_gemini(self, messages: List[Dict[str, str]]) -> str:
        """
        Format AutoGen messages for Gemini.
        
        Args:
            messages: List of message dictionaries from AutoGen
        
        Returns:
            Formatted prompt string for Gemini
        """
        formatted_messages = []
        
        # Extract system message if present
        system_message = "You are an AI assistant helping with talent matching."
        for message in messages:
            if message.get("role") == "system":
                system_message = message.get("content", system_message)
                break
        
        # Add system instruction
        formatted_messages.append(f"System instruction: {system_message}\n")
        
        # Format conversation
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            # Skip system messages (already handled)
            if role == "system":
                continue
                
            # Format based on role
            if role == "user":
                formatted_messages.append(f"User: {content}")
            elif role == "assistant":
                formatted_messages.append(f"Assistant: {content}")
            # Add 'function' role if needed
            
        # Join with newlines
        return "\n".join(formatted_messages)


def get_gemini_config_from_env() -> Optional[GeminiLLMConfig]:
    """
    Get Gemini configuration from environment variables.
    
    Returns:
        GeminiLLMConfig object if configured, None otherwise
    """
    api_key = os.getenv('GEMINI_API_KEY')
    model_name = os.getenv('GEMINI_MODEL_NAME', 'gemini-1.5-pro')
    
    if not api_key:
        logger.warning("No GEMINI_API_KEY found in environment variables. Gemini integration disabled.")
        return None
    
    return GeminiLLMConfig(api_key=api_key, model_name=model_name)


def main():
    """Test Gemini integration if run directly."""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get Gemini configuration
    gemini_config = get_gemini_config_from_env()
    
    if gemini_config:
        # Initialize Gemini
        gemini_llm_config = gemini_config.get_gemini_for_autogen()
        
        print(f"Gemini configuration successful!")
        print(f"Model: {gemini_config.model_name}")
        
        # Test a simple prompt
        try:
            model = genai.GenerativeModel(gemini_config.model_name)
            response = model.generate_content("Explain how to match candidates to jobs in 2 sentences.")
            print("\nTest response from Gemini:")
            print(response.text)
        except Exception as e:
            print(f"Error testing Gemini API: {str(e)}")
    else:
        print("Gemini API key not found. Please set GEMINI_API_KEY in your environment.")

if __name__ == "__main__":
    main()