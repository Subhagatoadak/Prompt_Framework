# __init__.py

from .Prompt_Framework import Prompt_Framework

# You can also import specific frameworks directly for convenience if needed
from .Prompt_Framework import (
    race_framework,
    care_framework,
    ape_framework,
    create_framework,
    tag_framework,
    creo_framework,
    rise_framework,
    pain_framework,
    coast_framework,
    roses_framework,
    react_framework
)

__all__ = [
    "Prompt_Framework",
    "race_framework",
    "care_framework",
    "ape_framework",
    "create_framework",
    "tag_framework",
    "creo_framework",
    "rise_framework",
    "pain_framework",
    "coast_framework",
    "roses_framework",
    "react_framework"
]
