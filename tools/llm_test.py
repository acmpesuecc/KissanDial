#!/usr/bin/env python3
"""
LLM Configuration Test and Management Utility

This script helps test and manage the LLM configuration for KissanDial.
It can test connections, list available models, and benchmark different models.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.llm_factory import (
    create_llm, 
    get_provider_info, 
    list_available_models, 
    test_llm_connection,
    GEMINI_MODELS
)

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_provider_info():
    """Display current provider configuration."""
    print_header("Current LLM Configuration")
    
    info = get_provider_info()
    print(f"Provider: {info['provider'].upper()}")
    print(f"Model: {info['model']}")
    print(f"API Key Configured: {'✓' if info['api_key_configured'] else '✗'}")
    
    if info['provider'] == 'gemini' and 'model_descriptions' in info:
        print(f"Model Description: {info['model_descriptions'].get(info['model'], 'Unknown model')}")

def print_available_models():
    """Display all available models for current provider."""
    print_header("Available Models")
    
    provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    models = list_available_models(provider)
    
    print(f"Available {provider.upper()} models:")
    for model, description in models.items():
        current = "→ " if model == get_provider_info()['model'] else "  "
        print(f"{current}{model}: {description}")

def test_connection():
    """Test the current LLM connection."""
    print_header("Connection Test")
    
    print("Testing LLM connection...")
    result = test_llm_connection()
    
    if result['success']:
        print("✓ Connection successful!")
        print(f"Provider: {result['provider']}")
        print(f"Model: {result['model']}")
        print(f"Test response: {result['response']}")
    else:
        print("✗ Connection failed!")
        print(f"Error: {result['error']}")

def benchmark_models():
    """Benchmark different models with a farmer-specific prompt."""
    print_header("Model Benchmark")
    
    # Farmer-specific test prompt
    test_prompt = """
    A farmer in Karnataka is asking: "My tomato plants are showing yellow leaves and the fruits are small. 
    The monsoon was delayed this year. What should I do to improve my crop yield?"
    
    Please provide a helpful response in 2-3 sentences.
    """
    
    provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    
    if provider == 'gemini':
        models_to_test = ['gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash']
    else:
        models_to_test = ['gpt-4o', 'gpt-4']
    
    results = {}
    
    for model in models_to_test:
        try:
            print(f"\nTesting {model}...")
            
            start_time = time.time()
            llm = create_llm(model_override=model)
            response = llm.complete(test_prompt)
            end_time = time.time()
            
            results[model] = {
                'response': str(response),
                'time': end_time - start_time,
                'success': True
            }
            
            print(f"✓ {model}: {results[model]['time']:.2f}s")
            
        except Exception as e:
            results[model] = {
                'error': str(e),
                'success': False
            }
            print(f"✗ {model}: {str(e)}")
    
    # Display results
    print_header("Benchmark Results")
    for model, result in results.items():
        if result['success']:
            print(f"\n{model} ({result['time']:.2f}s):")
            print("-" * 40)
            print(result['response'][:200] + "..." if len(result['response']) > 200 else result['response'])
        else:
            print(f"\n{model}: FAILED - {result['error']}")

def interactive_menu():
    """Interactive menu for testing LLM configuration."""
    while True:
        print_header("KissanDial LLM Test Utility")
        print("1. Show current configuration")
        print("2. List available models")
        print("3. Test connection")
        print("4. Benchmark models")
        print("5. Set Gemini model")
        print("6. Exit")
        
        choice = input("\nSelect an option (1-6): ").strip()
        
        if choice == '1':
            print_provider_info()
        elif choice == '2':
            print_available_models()
        elif choice == '3':
            test_connection()
        elif choice == '4':
            benchmark_models()
        elif choice == '5':
            set_gemini_model()
        elif choice == '6':
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")
        
        input("\nPress Enter to continue...")

def set_gemini_model():
    """Interactive Gemini model selection."""
    print_header("Set Gemini Model")
    
    if os.getenv('LLM_PROVIDER', '').lower() != 'gemini':
        print("LLM_PROVIDER is not set to 'gemini'. Please update your .env file first.")
        return
    
    print("Available Gemini models:")
    models = list(GEMINI_MODELS.keys())
    
    for i, model in enumerate(models, 1):
        current = " (current)" if model == get_provider_info()['model'] else ""
        print(f"{i}. {model}: {GEMINI_MODELS[model]}{current}")
    
    try:
        choice = int(input(f"\nSelect a model (1-{len(models)}): "))
        if 1 <= choice <= len(models):
            selected_model = models[choice - 1]
            
            # Update .env file
            env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
            
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    lines = f.readlines()
                
                # Update or add GEMINI_MODEL line
                updated = False
                for i, line in enumerate(lines):
                    if line.startswith('GEMINI_MODEL='):
                        lines[i] = f'GEMINI_MODEL={selected_model}\n'
                        updated = True
                        break
                
                if not updated:
                    lines.append(f'GEMINI_MODEL={selected_model}\n')
                
                with open(env_path, 'w') as f:
                    f.writelines(lines)
                
                print(f"✓ Model set to {selected_model}")
                print("Please restart the application to use the new model.")
            else:
                print("✗ .env file not found. Please create one first.")
        else:
            print("Invalid selection.")
    except ValueError:
        print("Invalid input. Please enter a number.")

def main():
    """Main function."""
    load_dotenv()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == 'info':
            print_provider_info()
        elif command == 'models':
            print_available_models()
        elif command == 'test':
            test_connection()
        elif command == 'benchmark':
            benchmark_models()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: info, models, test, benchmark")
    else:
        interactive_menu()

if __name__ == "__main__":
    main()
