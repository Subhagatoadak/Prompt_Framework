class Prompt_Framework:
    """
    A class for generating prompts based on various prompt engineering frameworks. 
    This allows easy switching between frameworks while maintaining common inputs.
    
    Attributes:
    context (str): A common context for all frameworks.
    output_type (str): Desired output type (optional).
    style (str): Desired output style (optional).
    role (str): Role for frameworks that require it (e.g., ROSE, RISE).
    """

    def __init__(self, context: str, output_type: str = None, style: str = None, role: str = None, *args, **kwargs):
        """
        Initializes the prompt framework with common parameters that are shared across multiple frameworks.
        
        Args:
        context (str): Common context to be used in all prompts.
        output_type (str, optional): Type of output (e.g., solution, answer, summary).
        style (str, optional): Desired style for the output (e.g., polite, formal).
        role (str, optional): Role for frameworks like ROSE or RISE that require a role parameter.
        """
        self.context = context
        self.output_type = output_type
        self.style = style
        self.role = role
        self.args = args
        self.kwargs = kwargs
        self.framework = None

    def switch_framework(self, framework: str):
        """
        Switches to a specific prompt engineering framework based on the user's choice.
        
        Args:
        framework (str): The framework to switch to (e.g., "costar", "care", "race").
        
        Raises:
        ValueError: If an invalid framework name is provided.
        """
        frameworks = {
            "costar": self.costar_framework,
            "care": self.care_framework,
            "race": self.race_framework,
            "ape": self.ape_framework,
            "create": self.create_framework,
            "tag": self.tag_framework,
            "creo": self.creo_framework,
            "rise": self.rise_framework,
            "pain": self.pain_framework,
            "coast": self.coast_framework,
            "roses": self.roses_framework,
            "react": self.react_framework,
        }
        
        framework = framework.lower()
        
        if framework in frameworks:
            self.framework = frameworks[framework]
        else:
            raise ValueError(f"Invalid framework: {framework}. Please choose a valid framework.")

    def generate_prompt(self, *args, **kwargs):
        """
        Generates a prompt based on the selected framework. The appropriate method for the 
        chosen framework is invoked, passing the framework-specific arguments.
        
        Args:
        *args: Arguments specific to the framework's method.
        **kwargs: Keyword arguments specific to the framework's method.
        
        Returns:
        str: The generated prompt based on the selected framework.
        
        Raises:
        ValueError: If no framework has been selected.
        """
        if self.framework is None:
            raise ValueError("No framework selected. Use 'switch_framework' to select a framework.")
        
        return self.framework(*args, **kwargs)

    def costar_framework(self, reasoning: str, *args, **kwargs):
        """
        Constructs a prompt using the CoStar framework.
        
        CoStar Framework requires reasoning as the primary component to guide the response generation.
        
        Args:
        reasoning (str): The reasoning or steps taken to reach the answer.
        
        Returns:
        str: The generated CoStar framework prompt.
        """
        prompt = f"Context: {self.context}\n"
        if self.output_type: prompt += f"Output Type: {self.output_type}\n"
        if self.style: prompt += f"Style: {self.style}\n"
        prompt += f"Reasoning: {reasoning}\n"
        prompt += "Task: Generate a response based on the above parameters."
        return prompt

    def care_framework(self, action: str, result: str, example: str, *args, **kwargs):
        """
        Constructs a prompt using the CARE framework.
        
        CARE Framework requires action, result, and example to guide the response generation.
        
        Args:
        action (str): Action needed based on the context.
        result (str): The expected result of the action.
        example (str): A sample example to demonstrate the expected output.
        
        Returns:
        str: The generated CARE framework prompt.
        """
        prompt = f"Context: {self.context}\n"
        if self.output_type: prompt += f"Output Type: {self.output_type}\n"
        if self.style: prompt += f"Style: {self.style}\n"
        prompt += f"Action: {action}\n"
        prompt += f"Expected Result: {result}\n"
        prompt += f"Example: {example}\n"
        prompt += "Task: Generate a response based on the above information."
        return prompt

    def race_framework(self, role: str, action: str, explanation: str, *args, **kwargs):
        """
        Constructs a prompt using the RACE framework.
        
        RACE Framework requires role, action, context, and explanation to guide the response generation.
        
        Args:
        role (str): The role of the responder.
        action (str): The action to be performed.
        explanation (str): The reasoning behind the action.
        
        Returns:
        str: The generated RACE framework prompt.
        """
        prompt = f"Role: {role}\n"
        prompt += f"Action: {action}\n"
        prompt += f"Context: {self.context}\n"
        prompt += f"Explanation: {explanation}\n"
        prompt += "Task: Generate a response based on the above parameters."
        return prompt

    def ape_framework(self, action: str, purpose: str, execution: str, *args, **kwargs):
        """
        Constructs a prompt using the APE framework.
        
        APE Framework requires action, purpose, and execution to guide the response generation.
        
        Args:
        action (str): The job to be done.
        purpose (str): The purpose or goal of the task.
        execution (str): The expected outcome or execution of the task.
        
        Returns:
        str: The generated APE framework prompt.
        """
        prompt = f"Action: {action}\n"
        prompt += f"Purpose: {purpose}\n"
        prompt += f"Execution: {execution}\n"
        prompt += "Task: Generate a response based on the above information."
        return prompt

    def create_framework(self, character: str, request: str, examples: str, adjustment: str, output_type: str, *args, **kwargs):
        """
        Constructs a prompt using the CREATE framework.
        
        CREATE Framework requires a character, request, examples, adjustment, and output type to guide the response generation.
        
        Args:
        character (str): The role or character to take in the task.
        request (str): The task to be performed.
        examples (str): Example outputs to follow.
        adjustment (str): Instructions to improve or better the task.
        output_type (str): The desired output format.
        
        Returns:
        str: The generated CREATE framework prompt.
        """
        prompt = f"Character: {character}\n"
        prompt += f"Request: {request}\n"
        prompt += f"Examples: {examples}\n"
        prompt += f"Adjustment: {adjustment}\n"
        prompt += f"Type of Output: {output_type}\n"
        prompt += "Task: Generate a response based on the above parameters."
        return prompt

    def tag_framework(self, task: str, action: str, goal: str, *args, **kwargs):
        """
        Constructs a prompt using the TAG framework.
        
        TAG Framework requires task, action, and goal to guide the response generation.
        
        Args:
        task (str): The task to be performed.
        action (str): The action to be taken for the task.
        goal (str): The end goal or objective of the task.
        
        Returns:
        str: The generated TAG framework prompt.
        """
        prompt = f"Task: {task}\n"
        prompt += f"Action: {action}\n"
        prompt += f"Goal: {goal}\n"
        prompt += "Task: Generate a response based on the above parameters."
        return prompt

    def creo_framework(self, context: str, request: str, explanation: str, outcome: str, *args, **kwargs):
        """
        Constructs a prompt using the CREO framework.
        
        CREO Framework requires context, request, explanation, and outcome to guide the response generation.
        
        Args:
        context (str): Background information for the task.
        request (str): The task to be performed.
        explanation (str): The explanation or reasoning for the task.
        outcome (str): The expected outcome of the task.
        
        Returns:
        str: The generated CREO framework prompt.
        """
        prompt = f"Context: {self.context}\n"
        prompt += f"Request: {request}\n"
        prompt += f"Explanation: {explanation}\n"
        prompt += f"Outcome: {outcome}\n"
        prompt += "Task: Generate a response based on the above parameters."
        return prompt

    def rise_framework(self, role: str, input_: str, steps: str, execution: str, *args, **kwargs):
        """
        Constructs a prompt using the RISE framework.
        
        RISE Framework requires role, input, steps, and execution to guide the response generation.
        
        Args:
        role (str): The role of the responder.
        input_ (str): Input data or context.
        steps (str): A list of steps to perform.
        execution (str): The outcome of the task.
        
        Returns:
        str: The generated RISE framework prompt.
        """
        prompt = f"Role: {role}\n"
        prompt += f"Input: {self.context}\n"
        prompt += f"Steps: {steps}\n"
        prompt += f"Execution: {execution}\n"
        prompt += "Task: Generate a response based on the above parameters."
        return prompt

    def pain_framework(self, problem: str, action: str, information: str, next_steps: str, *args, **kwargs):
        """
        Constructs a prompt using the PAIN framework.
        
        PAIN Framework requires problem, action, information, and next steps to guide the response generation.
        
        Args:
        problem (str): The problem to be addressed.
        action (str): The action to be taken to solve the problem.
        information (str): Additional information needed to understand the situation.
        next_steps (str): The next steps to proceed with after the action.
        
        Returns:
        str: The generated PAIN framework prompt.
        """
        prompt = f"Problem: {problem}\n"
        prompt += f"Action: {action}\n"
        prompt += f"Information: {information}\n"
        prompt += f"Next Steps: {next_steps}\n"
        prompt += "Task: Generate a response based on the above parameters."
        return prompt

    def coast_framework(self, context: str, objective: str, actions: str, scenario: str, task: str, *args, **kwargs):
        """
        Constructs a prompt using the COAST framework.
        
        COAST Framework requires context, objective, actions, scenario, and task to guide the response generation.
        
        Args:
        context (str): Background information.
        objective (str): The goal to be achieved.
        actions (str): The actions required to achieve the goal.
        scenario (str): The scenario or background context for the task.
        task (str): The task to be completed.
        
        Returns:
        str: The generated COAST framework prompt.
        """
        prompt = f"Context: {self.context}\n"
        prompt += f"Objective: {objective}\n"
        prompt += f"Actions: {actions}\n"
        prompt += f"Scenario: {scenario}\n"
        prompt += f"Task: {task}\n"
        prompt += "Task: Generate a response based on the above parameters."
        return prompt

    def roses_framework(self, role: str, objective: str, scenario: str, expected_solution: str, steps: str, *args, **kwargs):
        """
        Constructs a prompt using the ROSES framework.
        
        ROSES Framework requires role, objective, scenario, expected solution, and steps to guide the response generation.
        
        Args:
        role (str): The role of the responder.
        objective (str): The objective or result needed.
        scenario (str): The background information.
        expected_solution (str): The expected solution or result.
        steps (str): The steps needed to solve the problem.
        
        Returns:
        str: The generated ROSES framework prompt.
        """
        prompt = f"Role: {role}\n"
        prompt += f"Objective: {objective}\n"
        prompt += f"Scenario: {scenario}\n"
        prompt += f"Expected Solution: {expected_solution}\n"
        prompt += f"Steps: {steps}\n"
        prompt += "Task: Generate a response based on the above parameters."
        return prompt

    def react_framework(self, context: str, task: str, explanation: str, *args, **kwargs):
        """
        Constructs a prompt using the REACT framework.
        
        REACT Framework requires context, task, and explanation to guide the response generation.
        
        Args:
        context (str): Background information.
        task (str): The task to be completed.
        explanation (str): Explanation of the task or problem.
        
        Returns:
        str: The generated REACT framework prompt.
        """
        prompt = f"Context: {self.context}\n"
        prompt += f"Task: {task}\n"
        prompt += f"Explanation: {explanation}\n"
        prompt += "Task: Generate a response based on the above parameters."
        return prompt

