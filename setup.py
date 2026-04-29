"""
Setup script for sentiment-analysis package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sentiment-analysis",
    version="1.0.0",
    author="Uriel21900",
    author_email="uriel21900@example.com",
    description="Real-time sentiment analysis dashboard with FastAPI and Streamlit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "notebooks", "dashboards"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)
