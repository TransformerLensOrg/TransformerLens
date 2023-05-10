from setuptools import setup

setup(
    name="transformer_lens",
    version="0.1.0",
    packages=["transformer_lens"],
    license="LICENSE",
    description="An implementation of transformers tailored for mechanistic interpretability.",
    long_description=open("README.md").read(),
    install_requires=[
        "einops",
        "numpy",
        "torch",
        "datasets",
        "transformers",
        "tqdm",
        "pandas",
        "datasets",
        "wandb",
        "fancy_einsum",
        "rich",
        "accelerate",
        "typing-extensions",
    ],
    extras_require={"dev": ["pytest", "mypy", "pytest-cov"]},
)
