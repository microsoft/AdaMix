import setuptools

setuptools.setup(
    name="loralib",
    version="0.1.0",
    author="",
    author_email="",
    description="PyTorch implementation of low-rank adaptation (LoRA) and Adamix, a parameter-efficient approach to adapt a large pre-trained deep learning model which obtains performance better than full fine-tuning.",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)