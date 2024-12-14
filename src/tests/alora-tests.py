import pytest
from src.ai_assistant import AloraAssistant
from src.models.reasoning_engine import AloraReasoningEngine
from src.utils.knowledge_base import AloraKnowledgeBase

def test_alora_assistant_initialization():
    """Test that Alora Assistant initializes correctly."""
    assistant = AloraAssistant()
    assert isinstance(assistant.reasoning_engine, AloraReasoningEngine)
    assert isinstance(assistant.knowledge_base, AloraKnowledgeBase)

def test_process_input():
    """Test the process_input method returns a response."""
    assistant = AloraAssistant()
    response = assistant.process_input("Hello, Alora! How are you today?")
    assert isinstance(response, str)
    assert len(response) > 0

def test_reasoning_engine():
    """Test the Alora Reasoning Engine analysis method."""
    engine = AloraReasoningEngine()
    result = engine.analyze("Test input for Alora")
    assert isinstance(result, dict)
    assert "sentiment" in result
    assert "entities" in result
    assert "logical_inference" in result
    assert "complexity" in result

def test_knowledge_base():
    """Test the Alora Knowledge Base query method."""
    kb = AloraKnowledgeBase()
    result = kb.query("AI")
    assert isinstance(result, dict)
    assert len(result) > 0
