import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lm_eval",
    version="0.3.0",
    author="Leo Gao",
    author_email="lg@eleuther.ai",
    description="A framework for evaluating autoregressive language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EleutherAI/lm-evaluation-harness",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "datasets",
        "click>=7.1",
        "sqlitedict",
        "torch>=1.7",
        "transformers>=4.1",
        "sacrebleu==1.5.0",
        "lm_dataformat==0.0.20",
        "zstandard==0.15.2",
        "mock==4.0.3",
        "jieba==0.42.1",
        "pytest",
        "mosaicml"
        "jsonlines",
        "numexpr",
        "openai>=0.6.4",
        "pybind11>=2.6.2",
        "pycountry",
        "pytablewriter",
        "rouge-score>=0.0.4",
        "scikit-learn>=0.24.1",
        "sqlitedict",
        "tqdm-multiprocess",
        "zstandard",
    ],
    dependency_links=[
        # "https://github.com/google-research/bleurt/archive/b610120347ef22b494b6d69b4316e303f5932516.zip#egg=bleurt",
        "zstandard",
    ],
    extras_require={
        "dev": ["black", "flake8", "pre-commit", "pytest", "pytest-cov"],
        "multilingual": ["nagisa>=0.2.7", "jieba>=0.42.1"],
    },
)
