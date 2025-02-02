# **Prompt Framework**

**Prompt_Framework** is a Python package that provides a set of flexible frameworks for prompt engineering. It allows seamless interchangability between various frameworks such as **RACE**, **CARE**, **APE**, **CREATE**, **TAG**, **CREO**, **RISE**, **PAIN**, **COAST**, **ROSES**, and **REACT** to build sophisticated prompts for language models with different context and task-based structures.

---

## **Features**

- **Multiple Framework Support**: Includes popular prompt engineering frameworks like **RACE**, **CARE**, **APE**, **CREATE**, **TAG**, **CREO**, **RISE**, **PAIN**, **COAST**, **ROSES**, and **REACT**.
- **Seamless Switching**: Easily switch between frameworks without writing new prompt logic.
- **Customizable Prompts**: Add common parameters during initialization and use framework-specific arguments as needed.
- **Flexible Input**: Accepts various parameters, allowing customization of prompts for different use cases.

---

## **Installation**

To install **Prompt_Framework**, follow these steps:

### **1. Install via pip (for local development)**

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/yourusername/Prompt_Framework.git
cd Prompt_Framework
pip install -e .
```

## **Usage**

Once you’ve installed the package, you can easily import and start using the **Prompt_Framework** in your Python scripts.

### **1. Import the `PromptFramework` Class**

First, import the `PromptFramework` class into your Python script:

```python
from Prompt_Framework import Prompt_Framework

# Initialize with common parameters
prompt_tool = Prompt_Framework(context="Customer service inquiry.", output_type="solution", style="polite")
```

### **2. Switch to a Framework**

To generate a prompt for a specific framework, use the `switch_framework()` method. For instance, you can switch to the **RACE** framework:

```python
# Switch to the RACE framework and generate a prompt
prompt_tool.switch_framework("race")
race_prompt = prompt_tool.generate_prompt(
    role="Assistant",
    action="Provide customer service assistance",
    explanation="Provide a detailed and polite response based on the inquiry."
)

print(race_prompt)
```
### **Switch Between Frameworks**


You can easily switch between different frameworks as per your needs. For example, after generating a **RACE** prompt, you might want to switch to the **APE** framework:

```python
# Switch to APE framework and generate a prompt
prompt_tool.switch_framework("ape")
ape_prompt = prompt_tool.generate_prompt(
    action="Solve account issue",
    purpose="Help the user recover their account",
    execution="Provide detailed account recovery instructions."
)

print(ape_prompt)
```


### **Customizing Frameworks**


Frameworks come with specific requirements for the prompt parameters. For example:

#### **RACE Framework**

The **RACE** framework might require:
- **role**: The role of the assistant (e.g., "Assistant").
- **action**: What action the assistant should take (e.g., "Provide customer service assistance").
- **explanation**: The explanation of the task (e.g., "Provide a detailed and polite response based on the inquiry.").

```python
# Customizing RACE framework
prompt_tool.switch_framework("race")
race_prompt = prompt_tool.generate_prompt(
    role="Customer Service Representative",
    action="Assist with product inquiry",
    explanation="Provide helpful and polite product details in response to customer queries."
)

print(race_prompt)
```


## Frameworks Included
The following prompt engineering frameworks are included in the package:

### 1. RACE
Role: Specify role
Action: Mention action needed
Context: Provide background information
Explanation: Describe the outcome
### 2. CARE
Context: Provide background information
Action: Mention action needed
Result: State the goal
Example: Give example outputs
### 3. APE
Action: Define the job to be done
Purpose: State the goal
Execution: Describe the desired outcome
### 4. CREATE
Character: Specify the role
Request: Define the job to be done
Examples: Provide example outputs
Adjustment: Suggestions for improvement
Type of Output: Define the output format
Extras: Additional context
### 5. TAG
Task: Define the task
Action: Define job to be done
Goal: Describe the end goal
### 6. CREO
Context: Provide background information
Request: Define the job to be done
Explanation: Explain the task
Outcome: Describe the desired outcome
### 7. RISE
Role: Mention the role
Input: Provide context and instructions
Steps: Ask for step-by-step instructions
Execution: Describe the desired outcome
### 8. PAIN
Problem: Describe the problem
Action: Mention the action needed
Information: Request any necessary details
Next Steps: Ask for resources or next steps
### 9. COAST
Context: Provide background information
Objective: Define the goal
Actions: List actions required
Scenario: Describe the scenario
Task: Define the task to be completed
### 10. ROSES
Role: Define the role of the responder
Objective: State the expected result
Scenario: Provide background context
Expected Solution: Describe the expected solution
Steps: Ask for the steps to achieve the solution
### 11. REACT
Context: Provide background information
Task: Define the task to be completed
Explanation: Describe the task or problem

## API Reference
PromptFramework Class
Initialization
```python
prompt_tool = PromptFramework(context="Context information", output_type="desired_output", style="desired_style")
```
context (str): Background information for the prompt.
output_type (str): The expected output format (e.g., "solution", "steps").
style (str): Style of the output (e.g., "polite", "casual").
switch_framework(framework_name)

Switches to a specific framework.

framework_name (str): The name of the framework to switch to. Options include "race", "ape", "create", etc.

generate_prompt(*args, **kwargs)

Generates a prompt using the currently selected framework.

args: Framework-specific arguments.
kwargs: Additional keyword arguments for customization.

## Contributing
We welcome contributions! If you'd like to improve this package, feel free to fork the repository, make your changes, and submit a pull request. Please make sure to write tests for new features or bug fixes.

### Steps to Contribute:
Fork the repository
Clone your fork: git clone https://github.com/yourusername/Prompt_Framework.git
Create a new branch: git checkout -b feature-branch
Make your changes
Push your changes: git push origin feature-branch
Create a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.


