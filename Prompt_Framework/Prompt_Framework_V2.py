# prompt_framework.py
"""
Production-ready Prompt Framework module supporting 12 built-in prompt engineering frameworks,
custom framework registration, RL-based prompt tuning, single/few/zero-shot injection, prompt chaining,
chain-of-thought and advanced reasoning patterns, RAG integration, logging, validation, and sanitization.
"""
import json
import logging
import random
import re
import yaml
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field, ValidationError

# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------------------
# Exceptions
# ------------------------------
class PromptFrameworkError(Exception):
    """Custom exception class for Prompt Framework errors."""
    pass

# ------------------------------
# Shot Mode Enumeration
# ------------------------------
class ShotMode(str, Enum):
    ZERO = "zero"
    SINGLE = "single"
    FEW = "few"

# ------------------------------
# Validation & Sanitization
# ------------------------------
class PromptValidator:
    @staticmethod
    def validate(prompt: str, max_length: int = 2000) -> None:
        """Ensure prompt is within length and has no unfilled placeholders."""
        if len(prompt) > max_length:
            raise PromptFrameworkError(f"Prompt exceeds max length of {max_length} characters")
        if "{" in prompt or "}" in prompt:
            raise PromptFrameworkError("Prompt contains unfilled placeholders")

class PromptSanitizer:
    @staticmethod
    def sanitize(text: str) -> str:
        """Redact PII such as emails and phone numbers."""
        text = re.sub(r"\S+@\S+", "[REDACTED]", text)
        text = re.sub(r"\b\d{10}\b", "[REDACTED]", text)
        return text

# ------------------------------
# Reinforcement Learning Tuner
# ------------------------------
class PromptTuner:
    def __init__(self, epsilon: float = 0.1, min_epsilon: float = 0.01, decay: float = 0.99):
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        # stats: framework -> version -> (total_reward, count)
        self._stats: Dict[str, Dict[str, Tuple[float, int]]] = defaultdict(lambda: defaultdict(lambda: (0.0, 0)))

    def record_reward(self, framework: str, version: str, reward: float) -> None:
        """Record a user-provided reward for a given framework version."""
        total, count = self._stats[framework][version]
        self._stats[framework][version] = (total + reward, count + 1)
        logger.info(f"Recorded reward={reward:.2f} for {framework}@{version}")

    def select_version(self, framework: str, versions: List[str]) -> str:
        """Epsilon-greedy selection among available template versions."""
        if random.random() < self.epsilon:
            choice = random.choice(versions)
            logger.debug(f"Exploring {framework}@{choice} (epsilon={self.epsilon:.3f})")
            return choice
        # exploit best average reward
        averages = {
            v: (self._stats[framework][v][0] / self._stats[framework][v][1])
            for v in versions if self._stats[framework][v][1] > 0
        }
        if averages:
            best = max(averages, key=averages.get)
            logger.debug(f"Exploiting {framework}@{best} (epsilon={self.epsilon:.3f})")
            return best
        # fallback to latest semantic version
        return sorted(versions, key=lambda v: list(map(int, v.split('.'))))[-1]

    def decay_epsilon(self) -> None:
        """Decay epsilon after each tuning cycle to reduce exploration over time."""
        old = self.epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        logger.info(f"Epsilon decayed from {old:.3f} to {self.epsilon:.3f}")

# ------------------------------
# Template Manager
# ------------------------------
class TemplateManager:
    def __init__(self, tuner: Optional[PromptTuner] = None, config_path: Optional[Union[str, Path]] = None):
        self._registry: Dict[str, Dict[str, Callable]] = defaultdict(dict)
        self.tuner = tuner
        if config_path:
            self._load_from_config(Path(config_path))

    def _load_from_config(self, path: Path) -> None:
        """Load framework definitions from a JSON or YAML config file."""
        raw = path.read_text()
        data = yaml.safe_load(raw) if path.suffix in ('.yaml', '.yml') else json.loads(raw)
        for framework, versions in data.items():
            for version, builder_path in versions.items():
                module_name, func_name = builder_path.rsplit(':', 1)
                module = __import__(module_name, fromlist=[func_name])
                builder = getattr(module, func_name)
                self.register(framework, version, builder)
        logger.info(f"Loaded templates from {path}")

    def register(self, framework: str, version: str, builder: Callable) -> None:
        """Register a prompt-builder function under a framework name and version."""
        self._registry[framework.lower()][version] = builder
        logger.info(f"Registered {framework}@{version}")

    def add_custom_framework(self, framework: str, version: str, builder: Callable) -> None:
        """Add a user-defined framework and builder function."""
        self.register(framework, version, builder)
        logger.info(f"Custom framework added: {framework}@{version}")

    def get(self, framework: str, version: Optional[str] = None) -> Tuple[Callable, str]:
        """Retrieve the builder and chosen version for a framework."""
        key = framework.lower()
        versions = list(self._registry.get(key, {}))
        if not versions:
            raise PromptFrameworkError(f"No templates found for framework '{framework}'")
        chosen = version or (
            self.tuner.select_version(key, versions) if self.tuner else sorted(versions)[-1]
        )
        builder = self._registry[key].get(chosen)
        if builder is None:
            raise PromptFrameworkError(f"Framework '{framework}' version '{chosen}' not registered")
        logger.debug(f"Using {framework}@{chosen}")
        return builder, chosen

# ------------------------------
# Parameter Models for Frameworks
# ------------------------------
class BaseParams(BaseModel):
    """Base class for all framework parameter models."""
    pass

class CoSTARParams(BaseParams): reasoning: str = Field(..., description="Chain-of-thought reasoning steps.")
class CAREParams(BaseParams): action: str = Field(...); result: str = Field(...); example: str = Field(...)
class RACEParams(BaseParams): role: str = Field(...); action: str = Field(...); explanation: str = Field(...)
class APEParams(BaseParams): action: str = Field(...); purpose: str = Field(...); execution: str = Field(...)
class CREATEParams(BaseParams):
    character: str = Field(...); request: str = Field(...)
    examples: str = Field(...); adjustment: str = Field(...); output_type: str = Field(...)
class TAGParams(BaseParams): task: str = Field(...); action: str = Field(...); goal: str = Field(...)
class CREOParams(BaseParams): request: str = Field(...); explanation: str = Field(...); outcome: str = Field(...)
class RISEParams(BaseParams): role: str = Field(...); steps: str = Field(...); execution: str = Field(...)
class PAINParams(BaseParams): 
    problem: str = Field(...); action: str = Field(...)
    information: str = Field(...); next_steps: str = Field(...)
class COASTParams(BaseParams):
    objective: str = Field(...); actions: str = Field(...)
    scenario: str = Field(...); task: str = Field(...)
class ROSESParams(BaseParams):
    role: str = Field(...); objective: str = Field(...)
    scenario: str = Field(...); expected_solution: str = Field(...); steps: str = Field(...)
class REACTParams(BaseParams): task: str = Field(...); explanation: str = Field(...)

# ------------------------------
# Template Registration Decorator
# ------------------------------
_tuner = PromptTuner()
_template_manager = TemplateManager(tuner=_tuner)

def register_template(framework: str, version: str = "1.0"):
    """Decorator to register a prompt-builder under a framework name and version."""
    def decorator(fn: Callable) -> Callable:
        _template_manager.register(framework, version, fn)
        return fn
    return decorator

# ------------------------------
# Builder Functions for 12 Frameworks
# ------------------------------
@register_template("costar")
def costar_v1(p: CoSTARParams, ctx: str, out: Optional[str], style: Optional[str]) -> str:
    return (
        f"Context: {ctx}\n" +
        (f"Output: {out}\n" if out else "") +
        (f"Style: {style}\n" if style else "") +
        f"Reasoning: {p.reasoning}\nTask: Provide a detailed answer."
    )

@register_template("care")
def care_v1(p: CAREParams, ctx: str, out: Optional[str], style: Optional[str]) -> str:
    return (
        f"Context: {ctx}\n" +
        (f"Output: {out}\n" if out else "") +
        (f"Style: {style}\n" if style else "") +
        f"Action: {p.action}\nExpected Result: {p.result}\nExample: {p.example}\nTask: Respond accordingly."
    )

@register_template("race")
def race_v1(p: RACEParams, ctx: str, out: Optional[str], style: Optional[str]) -> str:
    return (
        f"Role: {p.role}\n"
        f"Action: {p.action}\n"
        f"Context: {ctx}\n"
        f"Explanation: {p.explanation}\nTask: Provide a response."
    )

@register_template("ape")
def ape_v1(p: APEParams, ctx: str, out: Optional[str], style: Optional[str]) -> str:
    return (
        f"Action: {p.action}\n"
        f"Purpose: {p.purpose}\n"
        f"Execution: {p.execution}\n"
        "Task: Provide a response."
    )

@register_template("create")
def create_v1(p: CREATEParams, ctx: str, out: Optional[str], style: Optional[str]) -> str:
    return (
        f"Character: {p.character}\n",
        f"Request: {p.request}\n",
        f"Examples: {p.examples}\n",
        f"Adjustment: {p.adjustment}\n",
        f"Output Type: {p.output_type}\n",
        (f"Style: {style}\n" if style else "") +
        "Task: Provide a response."
    )

@register_template("tag")
def tag_v1(p: TAGParams, ctx: str, out: Optional[str], style: Optional[str]) -> str:
    return (
        f"Task: {p.task}\n"
        f"Action: {p.action}\n"
        f"Goal: {p.goal}\n"
        "Task: Provide a response."
    )

@register_template("creo")
def creo_v1(p: CREOParams, ctx: str, out: Optional[str], style: Optional[str]) -> str:
    return (
        f"Context: {ctx}\n"
        f"Request: {p.request}\n"
        f"Explanation: {p.explanation}\n"
        f"Outcome: {p.outcome}\n"
        "Task: Provide a response."
    )

@register_template("rise")
def rise_v1(p: RISEParams, ctx: str, out: Optional[str], style: Optional[str]) -> str:
    return (
        f"Role: {p.role}\n"
        f"Steps: {p.steps}\n"
        f"Input: {ctx}\n"
        f"Execution: {p.execution}\n"
        "Task: Provide a response."
    )

@register_template("pain")
def pain_v1(p: PAINParams, ctx: str, out: Optional[str], style: Optional[str]) -> str:
    return (
        f"Problem: {p.problem}\n"
        f"Action: {p.action}\n"
        f"Information: {p.information}\n"
        f"Next Steps: {p.next_steps}\n"
        "Task: Provide a response."
    )

@register_template("coast")
def coast_v1(p: COASTParams, ctx: str, out: Optional[str], style: Optional[str]) -> str:
    return (
        f"Context: {ctx}\n"
        f"Objective: {p.objective}\n"
        f"Actions: {p.actions}\n"
        f"Scenario: {p.scenario}\n"
        f"Task: {p.task}\n"
        "Provide a response."
    )

@register_template("roses")
def roses_v1(p: ROSESParams, ctx: str, out: Optional[str], style: Optional[str]) -> str:
    return (
        f"Role: {p.role}\n"
        f"Objective: {p.objective}\n"
        f"Scenario: {p.scenario}\n"
        f"Expected Solution: {p.expected_solution}\n"
        f"Steps: {p.steps}\n"
        "Task: Provide a response."
    )

@register_template("react")
def react_v1(p: REACTParams, ctx: str, out: Optional[str], style: Optional[str]) -> str:
    return (
        f"Context: {ctx}\n"
        f"Task: {p.task}\n"
        f"Explanation: {p.explanation}\n"
        "Task: Provide a response."
    )

# ------------------------------
# Core PromptFramework API
# ------------------------------
class PromptFramework:
    def __init__(
        self,
        context: str,
        output_type: Optional[str] = None,
        style: Optional[str] = None,
        model_name: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 512,
        stop_sequences: Optional[List[str]] = None,
        examples: Optional[List[str]] = None,
        retriever: Optional[Callable[[str], str]] = None,
        shot_mode: ShotMode = ShotMode.FEW,
        thought_pattern: Optional[str] = None,
    ):
        self.context = context
        self.output_type = output_type
        self.style = style
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop_sequences = stop_sequences or []
        self.examples = examples or []
        self.retriever = retriever
        self.shot_mode = shot_mode
        self.thought_pattern = thought_pattern
        self._framework: Optional[str] = None
        self._version: Optional[str] = None
        self._builder: Optional[Callable] = None

    def add_custom_framework(self, name: str, version: str, builder: Callable) -> None:
        _template_manager.add_custom_framework(name, version, builder)

    def chain_frameworks(self, sequence: List[Tuple[str, BaseModel, Optional[str]]]) -> List[str]:
        return [self._invoke(fr, params, ver) for fr, params, ver in sequence]

    def switch_framework(self, name: str, version: Optional[str] = None) -> None:
        builder, ver = _template_manager.get(name, version)
        self._builder = builder
        self._framework = name
        self._version = ver

    def generate_prompt(self, params: BaseModel) -> str:
        if not self._builder:
            raise PromptFrameworkError("No framework selected. Call switch_framework first.")
        ctx = self._prepare_context()
        prefix = self._prepare_prefix()
        raw = self._builder(params, ctx, self.output_type, self.style)
        prompt = PromptSanitizer.sanitize(f"{prefix}{raw}")
        PromptValidator.validate(prompt)
        logger.info(f"Generated prompt for {self._framework}@{self._version}")
        return prompt

    def record_reward(self, reward: float) -> None:
        if not self._framework:
            raise PromptFrameworkError("No framework selected to record reward.")
        _tuner.record_reward(self._framework, self._version or 'latest', reward)
        _tuner.decay_epsilon()

    def _invoke(self, framework: str, params: BaseModel, version: Optional[str]) -> str:
        self.switch_framework(framework, version)
        return self.generate_prompt(params)

    def _prepare_context(self) -> str:
        ctx = self.context
        if self.retriever:
            docs = self.retriever(self.context)
            ctx = f"{docs}\n{ctx}"
        if self.shot_mode == ShotMode.FEW:
            shot_txt = "".join(f"Example: {e}\n" for e in self.examples)
            ctx = f"{shot_txt}{ctx}"
        elif self.shot_mode == ShotMode.SINGLE and self.examples:
            ctx = f"Example: {self.examples[0]}\n{ctx}"
        return ctx

    def _prepare_prefix(self) -> str:
        if self.thought_pattern == 'cot':
            return "Let's think step-by-step before answering.\n"
        if self.thought_pattern == 'self-ask':
            return "Let's ask sub-questions before answering.\n"
        if self.thought_pattern == 'tree':
            return "Let's explore reasoning branches in a tree-of-thought format.\n"
        return ""

# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    # Initialize framework with single-shot and CoT
    pf = PromptFramework(
        context="You are an AI assistant.",
        examples=["Example: foo"],
        shot_mode=ShotMode.SINGLE,
        thought_pattern='cot'
    )
    # Generate prompts for each built-in framework
    frameworks = [
        ("costar", CoSTARParams(reasoning="Step 1 then 2")),
        ("care", CAREParams(action="Act", result="Res", example="Ex")),
        ("race", RACEParams(role="Role", action="Act", explanation="Why")),
        ("ape", APEParams(action="Act", purpose="Purpose", execution="Exec")),
        ("create", CREATEParams(character="Char", request="Req", examples="Exs", adjustment="Adj", output_type="Text")),
        ("tag", TAGParams(task="Task", action="Act", goal="Goal")),
        ("creo", CREOParams(request="Req", explanation="Expl", outcome="Out")),
        ("rise", RISEParams(role="Role", steps="S1,S2", execution="Exec")),
        ("pain", PAINParams(problem="Prob", action="Act", information="Info", next_steps="Next")),
        ("coast", COASTParams(objective="Obj", actions="Acts", scenario="Scen", task="Task")),
        ("roses", ROSESParams(role="Role", objective="Obj", scenario="Scen", expected_solution="Sol", steps="S1")),
        ("react", REACTParams(task="Task", explanation="Expl"))
    ]
    for name, params in frameworks:
        pf.switch_framework(name)
        prompt = pf.generate_prompt(params)
        print(f"---{name}---\n{prompt}\n")
    # Simulate feedback
    pf.record_reward(0.9)

