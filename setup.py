"""
Setup script for crypto_backtesting package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Cryptocurrency Backtesting Framework"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="crypto-backtesting",
    version="1.0.0",
    author="Crypto Backtesting Team",
    author_email="team@crypto-backtesting.com",
    description="A comprehensive framework for backtesting cryptocurrency trading strategies",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/crypto-backtesting/crypto-backtesting",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "visualization": [
            "plotly>=5.0",
            "dash>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crypto-backtest=crypto_backtesting.cli.main:main",
            "crypto-collect-data=crypto_backtesting.cli.collect_data:main",
            "crypto-find-pairs=crypto_backtesting.cli.find_pairs:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)