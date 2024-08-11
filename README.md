# Hand on LLM

## Roadmap

### [1. The LLM architecture](./1.%20The%20LLM%20architecture)

* **High-level view**: Revisit the encoder-decoder Transformer architecture, and more specifically the decoder-only GPT architecture, which is used in every modern LLM.

* **Tokenization**: Understand how to convert raw text data into a format that the model can understand, which involves splitting the text into tokens (usually words or subwords).
* **Attention mechanisms**: Grasp the theory behind attention mechanisms, including self-attention and scaled dot-product attention, which allows the model to focus on different parts of the input when producing an output.
* **Text generation**: Learn about the different ways the model can generate output sequences. Common strategies include greedy decoding, beam search, top-k sampling, and nucleus sampling.

### [2. Building an instruction dataset](./2.%20Building%20an%20instruction%20dataset)

* **Alpaca-like dataset**: Generate synthetic data from scratch with the OpenAI API (GPT). You can specify seeds and system prompts to create a diverse dataset.
* **Advanced techniques**: Learn how to improve existing datasets with Evol-Instruct, how to generate high-quality synthetic data like in the Orca and phi-1 papers.
* **Filtering data**: Traditional techniques involving regex, removing near-duplicates, focusing on answers with a high number of tokens, etc.
* **Prompt templates**: There's no true standard way of formatting instructions and answers, which is why it's important to know about the different chat templates, such as ChatML, Alpaca, etc.

### 3. Pre-traing models



### 4. Supervised Fine-Tuning



### 5. RLHF



### 6. Evaluation



### 7. Quantization



### 8. New Trends



## Reference

* [llm-course](https://github.com/mlabonne/llm-course)