# tests/test_prompt_framework.py

import pytest
from .Prompt_Framework import PromptFramework

def test_initialization():
    # Test initialization with common parameters
    prompt_tool = PromptFramework(context="Customer service inquiry.", output_type="solution", style="polite")
    assert prompt_tool.context == "Customer service inquiry."
    assert prompt_tool.output_type == "solution"
    assert prompt_tool.style == "polite"

def test_switch_to_race_framework():
    prompt_tool = PromptFramework(context="Customer service inquiry.", output_type="solution", style="polite")
    
    # Switch to RACE framework
    prompt_tool.switch_framework("race")
    race_prompt = prompt_tool.generate_prompt(
        role="Customer Service Representative",
        action="Assist with product inquiry",
        explanation="Provide helpful and polite product details in response to customer queries."
    )
    
    # Validate the generated prompt
    assert "Customer Service Representative" in race_prompt
    assert "Assist with product inquiry" in race_prompt
    assert "Provide helpful and polite product details" in race_prompt

def test_switch_to_ape_framework():
    prompt_tool = PromptFramework(context="Customer service inquiry.", output_type="solution", style="polite")
    
    # Switch to APE framework
    prompt_tool.switch_framework("ape")
    ape_prompt = prompt_tool.generate_prompt(
        action="Solve account issue",
        purpose="Help the user recover their account",
        execution="Provide detailed account recovery instructions."
    )
    
    # Validate the generated prompt
    assert "Solve account issue" in ape_prompt
    assert "Help the user recover their account" in ape_prompt
    assert "Provide detailed account recovery instructions" in ape_prompt

def test_invalid_framework():
    prompt_tool = PromptFramework(context="Customer service inquiry.", output_type="solution", style="polite")
    
    # Try to switch to an invalid framework
    with pytest.raises(ValueError):
        prompt_tool.switch_framework("invalid_framework")
