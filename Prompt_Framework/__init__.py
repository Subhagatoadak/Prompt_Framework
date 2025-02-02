# __init__.py

from .prompt_framework import PromptFramework

# You can also import specific frameworks directly for convenience if needed
from .prompt_framework import (
    RACE,
    CARE,
    APE,
    CREATE,
    TAG,
    CREO,
    RISE,
    PAIN,
    COAST,
    ROSES,
    REACT
)

__all__ = [
    "PromptFramework",
    "RACE",
    "CARE",
    "APE",
    "CREATE",
    "TAG",
    "CREO",
    "RISE",
    "PAIN",
    "COAST",
    "ROSES",
    "REACT"
]
