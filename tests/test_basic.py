import pytest
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_import():
    """Test that we can import Flask and basic modules"""
    from flask import Flask
    assert Flask is not None

def test_twilio_import():
    """Test that we can import Twilio modules"""
    from twilio.twiml.voice_response import VoiceResponse
    assert VoiceResponse is not None

def test_pandas_import():
    """Test that we can import pandas"""
    import pandas as pd
    assert pd is not None

def test_app_structure():
    """Test that the app directory structure is correct"""
    app_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app')
    assert os.path.exists(app_dir)
    assert os.path.exists(os.path.join(app_dir, 'agent.py'))
    assert os.path.exists(os.path.join(app_dir, '__init__.py'))
