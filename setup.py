from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="markthat",
    version="0.1.0",
    author="",
    author_email="",
    description="A Python module for converting images to markdown using multimodal LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Flopsky/markthat",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "jinja2>=3.0.0",
    ],
    extras_require={
        "gemini": ["google-generativeai>=0.3.0"],
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.5.0"],
        "mistral": ["mistralai>=0.0.1"],
        "pdf": ["pymupdf>=1.20.0"],
        "all": [
            "google-generativeai>=0.3.0",
            "openai>=1.0.0",
            "anthropic>=0.5.0", 
            "mistralai>=0.0.1",
            "pymupdf>=1.20.0"
        ]
    }
) 