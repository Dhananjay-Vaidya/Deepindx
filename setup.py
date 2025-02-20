from setuptools import setup, find_packages

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="deepindx",
    version="0.1.0",
    author="Dhananjay Vaidya",
    author_email="dhananjayvaidya4154@gmail.com",
    description="DeepindX: An advanced deep learning library featuring NAS, Transfer Learning, Explainability, and Compression.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dhananjay-Vaidya/NeuroX",
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "tensorflow>=2.10.0",
        "numpy>=1.23.0",
        "scipy>=1.9.3",
        "pandas>=1.5.3",
        "scikit-learn>=1.1.3",
        "matplotlib>=3.6.2",
        "seaborn>=0.12.2",
        "shap>=0.41.0",
        "lime>=0.2.0.1",
        "captum>=0.5.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.1",
        "torchsummary>=1.5.1",
        "tensorflow-model-optimization>=0.7.4",
        "ray[tune]>=2.4.0",
        "optuna>=3.2.0",
        "wandb>=0.16.1",
        "tensorboard>=2.12.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.2.2",
            "black>=23.1.0",
            "flake8>=6.0.0",
            "sphinx>=5.3.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "docs": [
            "sphinx>=5.3.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "deepindx-nas=deepindx.nas.cli:main",
            "deepindx-transfer=deepindx.transfer.cli:main",
            "deepindx-explainer=deepindx.explainer.cli:main",
            "deepindx-compression=deepindx.compression.cli:main",
        ],
    },
    project_urls={
        "Documentation": "https://github.com/Dhananjay-Vaidya/NeuroX/wiki",
        "Source": "https://github.com/Dhananjay-Vaidya/NeuroX",
        "Tracker": "https://github.com/Dhananjay-Vaidya/NeuroX/issues",
    },
)
