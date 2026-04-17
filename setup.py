"""Install helper for the SAT reference implementation."""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
long_description = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""


setup(
    name="sat-aamas",
    version="0.1.0",
    description="Sequential Agent Tuning (SAT) - official implementation (AAMAS 2026).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yi Xie, Yangyang Xu, Yi Fan, Bo Liu",
    url="https://github.com/Yydc/SAT-AAMAS",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(exclude=("tests", "manuscripts", "scripts", "configs")),
    include_package_data=True,
    install_requires=[
        "torch>=2.1,<3.0",
        "transformers>=4.44,<5.0",
        "accelerate>=0.30",
        "numpy>=1.24,<3.0",
        "pyyaml>=6.0",
        "sentencepiece>=0.1.99",
        "tokenizers>=0.19",
        "tqdm>=4.66",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
