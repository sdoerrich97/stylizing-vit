from setuptools import setup, find_packages

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    # Replace relative paths with absolute URLs for PyPI
    long_description = long_description.replace("](examples/", "](https://github.com/sdoerrich97/stylizing-vit/blob/main/examples/")
    long_description = long_description.replace("](assets/", "](https://raw.githubusercontent.com/sdoerrich97/stylizing-vit/main/assets/")

setup(
    name="stylizing-vit",
    version="1.0.0",
    description="Official implementation of 'Stylizing ViT' for robust style transfer.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sebastian Doerrich",
    author_email="sebastian.doerrich@uni-bamberg.de",
    url="https://github.com/sdoerrich97/stylizing-vit",
    license="Apache-2.0",
    packages=find_packages(include=["stylizing_vit", "stylizing_vit.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.5.1",
        "timm>=1.0.14",
        "numpy>=2.2.2",
        "Pillow>=11.1.0",
        "huggingface_hub>=0.33.2",
        "scipy>=1.15.1",
        "torchvision>=0.20.1",
        "safetensors>=0.5.2",
    ],
    extras_require={
        "train": [
            "accelerate>=1.8.1",
            "wandb>=0.19.4",
            "torchmetrics>=1.6.1",
            "lpips>=0.1.4",
            "pandas>=2.2.3",
            "scikit-learn>=1.6.1",
            "medmnist>=3.0.1",
            "medmnistc>=0.1.0",
            "tqdm>=4.67.1",
            "PyYAML>=6.0.2",
            "opencv-python>=4.11.0.86",
            "scikit-image>=0.23.2",
            "matplotlib>=3.10.3",
            "albumentations>=2.0.8",
        ],
        "dev": [
            "pytest",
            "black",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
