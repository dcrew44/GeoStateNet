from setuptools import setup, find_packages

setup(
    name="state_classifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "wandb>=0.12.0",
        "tqdm>=4.60.0",
        "pyyaml>=6.0",
    ],
)
