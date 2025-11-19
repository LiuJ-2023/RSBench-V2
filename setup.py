"""
Setup script for RSBench-V3 library.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rsbench-v3",
    version="1.0.0",
    author="RSBench Team",
    author_email="support@rsbench-v3.com",
    description="Multi-Objective Recommender System Benchmark with LLM Integration",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/rsbench-v3",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
        "plotting": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rsbench=cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "recommender-systems",
        "multi-objective-optimization",
        "evolutionary-algorithms",
        "large-language-models",
        "machine-learning",
        "artificial-intelligence",
        "nsga-ii",
        "pareto-optimization",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-repo/rsbench-v3/issues",
        "Source": "https://github.com/your-repo/rsbench-v3",
        "Documentation": "https://rsbench-v3.readthedocs.io/",
    },
)