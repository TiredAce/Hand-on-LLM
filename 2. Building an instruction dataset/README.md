# 2. Building an instruction dataset

## Outline

* **Alpaca-like dataset**: Generate synthetic data from scratch with the OpenAI API (GPT). You can specify seeds and system prompts to create a diverse dataset.
* **Advanced techniques**: Learn how to improve existing datasets with Evol-Instruct, how to generate high-quality synthetic data like in the Orca and phi-1 papers.
* **Filtering data**: Traditional techniques involving regex, removing near-duplicates, focusing on answers with a high number of tokens, etc.
* **Prompt templates**: There's no true standard way of formatting instructions and answers, which is why it's important to know about the different chat templates, such as ChatML, Alpaca, etc.

## Hand on instructions 

* [Preparing a Dataset for Instruction tuning](./Preparing%20a%20Dataset%20for%20Instruction%20tuning/Preparing%20a%20Dataset%20for%20Instruction%20tuning.ipynb): Exploration of the Alpaca and Alpaca-GPT4 datasets and how to format them.