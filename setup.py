from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent

# Read the contents of your README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="Prompt_Framework",
    version="0.1",
    packages=find_packages(),
    author="Subhagato Adak",
    author_email="subhagato.adak@gmail.com",
    description="Prompt_Framework is a Python package that provides a set of flexible frameworks for prompt engineering. It allows seamless interchangability between various frameworks",
    long_description=long_description,                        # ← your README
    long_description_content_type="text/markdown",            # ← or "text/x-rst"
    url="https://github.com/Subhagatoadak/Prompt_Framework",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
      # your runtime deps
    ],
)
