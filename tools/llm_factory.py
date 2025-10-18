"""
LLM Factory for dynamic LLM selection based on environment configuration.
Supports both OpenAI and Google Gemini with latest models.
"""

import os
import google.generativeai as genai
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Available Gemini models (from latest API response)
GEMINI_MODELS = {
    'gemini-2.0-flash-exp': 'Gemini 2.0 Flash (Experimental) - Latest multimodal model',
    'gemini-1.5-flash': 'Gemini 1.5 Flash - Fast and efficient',
    'gemini-1.5-flash-8b': 'Gemini 1.5 Flash 8B - Smaller, faster model',
    'gemini-1.5-pro': 'Gemini 1.5 Pro - High capability model',
    'gemini-pro': 'Gemini Pro - Original pro model',
    'gemini-pro-vision': 'Gemini Pro Vision - Vision-enabled model'
}

# Default models for each provider
DEFAULT_MODELS = {
    'openai': 'gpt-4o',
    'gemini': 'gemini-2.0-flash-exp'  # Using latest Gemini 2.0 as default
}

def create_llm(model_override=None):
    """
    Create and return an LLM instance based on the LLM_PROVIDER environment variable.
    
    Args:
        model_override (str, optional): Override the default model selection
    
    Returns:
        LLM instance (OpenAI or Gemini)
    
    Raises:
        ValueError: If LLM_PROVIDER is not supported or required API keys are missing
    """
    llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    
    if llm_provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        model = model_override or os.getenv('OPENAI_MODEL') or DEFAULT_MODELS['openai']
        
        logging.info(f"Creating OpenAI LLM with model: {model}")
        return OpenAI(
            temperature=0.1,  # Slightly increased for more natural responses
            model=model,
            api_key=api_key
        )
    
    elif llm_provider == 'gemini':
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Configure the Gemini API  
        genai.configure(api_key=api_key)
        
        model = model_override or os.getenv('GEMINI_MODEL') or DEFAULT_MODELS['gemini']
        
        # Validate model is in available list
        if model not in GEMINI_MODELS:
            logging.warning(f"Model {model} not in known model list. Available models: {list(GEMINI_MODELS.keys())}")
        
        logging.info(f"Creating Gemini LLM with model: {model}")
        return Gemini(
            model=model,
            temperature=0.1,  # Slightly increased for more natural responses
            api_key=api_key
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}. Supported providers: 'openai', 'gemini'")

def get_provider_info():
    """
    Get information about the current LLM provider configuration.
    
    Returns:
        dict: Information about the current provider
    """
    llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    
    info = {
        'provider': llm_provider,
        'api_key_configured': False,
        'model': None,
        'available_models': []
    }
    
    if llm_provider == 'openai':
        info['api_key_configured'] = bool(os.getenv('OPENAI_API_KEY'))
        info['model'] = os.getenv('OPENAI_MODEL') or DEFAULT_MODELS['openai']
        info['available_models'] = ['gpt-4o', 'gpt-4', 'gpt-3.5-turbo']
    elif llm_provider == 'gemini':
        info['api_key_configured'] = bool(os.getenv('GEMINI_API_KEY'))
        info['model'] = os.getenv('GEMINI_MODEL') or DEFAULT_MODELS['gemini']
        info['available_models'] = list(GEMINI_MODELS.keys())
        info['model_descriptions'] = GEMINI_MODELS
    
    return info

def list_available_models(provider=None):
    """
    List all available models for a given provider or current provider.
    
    Args:
        provider (str, optional): Provider to list models for ('openai' or 'gemini')
    
    Returns:
        dict: Available models with descriptions
    """
    if provider is None:
        provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    
    if provider == 'gemini':
        return GEMINI_MODELS
    elif provider == 'openai':
        return {
            'gpt-4o': 'GPT-4 Omni - Latest multimodal model',
            'gpt-4': 'GPT-4 - High capability model',
            'gpt-3.5-turbo': 'GPT-3.5 Turbo - Fast and efficient'
        }
    else:
        return {}

def test_llm_connection():
    """
    Test the current LLM configuration by making a simple API call.
    
    Returns:
        dict: Test results with success status and details
    """
    try:
        llm = create_llm()
        provider_info = get_provider_info()
        
        # Simple test prompt
        test_prompt = "Say 'Hello from KissanDial!' in one sentence."
        response = llm.complete(test_prompt)
        
        return {
            'success': True,
            'provider': provider_info['provider'],
            'model': provider_info['model'],
            'response': str(response),
            'message': f"Successfully connected to {provider_info['provider']} with model {provider_info['model']}"
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f"Failed to connect to LLM: {str(e)}"
        }
