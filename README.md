# LLM From Scratch - Study Repository

A comprehensive study repository documenting the journey of building Large Language Models (LLMs) from scratch. This project explores the fundamental concepts and techniques required to understand and implement modern language models like GPT.

## Project Overview

This repository contains structured learning materials and hands-on code implementations covering all essential stages of LLM development. The focus is on understanding the theoretical foundations while implementing practical solutions.

## Repository Structure

```
LLM-From-Scratch/
├── stage-1/
│   └── text-generation/
│       ├── tokenizing-text.ipynb           # Text tokenization fundamentals
│       ├── byte_pair_encoding.ipynb        # BPE tokenization algorithm
│       ├── word_vec.ipynb                  # Word embeddings and embeddings layer
│       ├── positional_embeddings.ipynb     # Positional encoding for transformers
│       ├── the-verdict.txt                 # Sample text data for training
│       └── BY_MYSELF/                      # Personal implementation folder
│           ├── tokenizing_myself.ipynb     # Custom tokenization implementation
│           ├── Computer_Network.txt        # Reference material
│           └── Operating_System.txt        # Reference material
└── README.md                               # This file
```

## Topics Covered in Stage 1: Text Generation

### 1. **Tokenizing Text** (`tokenizing-text.ipynb`)
   - Understanding text tokenization concepts
   - Basic tokenization patterns and regex-based splitting
   - Converting raw text into tokens
   - Preparing text data for model input
   - Working with the `the-verdict.txt` dataset

### 2. **Byte Pair Encoding (BPE)** (`byte_pair_encoding.ipynb`)
   - Understanding subword tokenization
   - Implementation of the Byte Pair Encoding algorithm
   - Advantages of BPE over character and word-level tokenization
   - Using pre-trained tokenizers (GPT-2 tokenizer via `tiktoken`)
   - Encoding and decoding text with BPE

### 3. **Word Embeddings** (`word_vec.ipynb`)
   - Introduction to word embeddings and vector representations
   - Word2Vec models and pre-trained embeddings
   - PyTorch Embedding layer implementation
   - Creating embedding matrices for vocabulary
   - Understanding semantic meaning in vector space
   - Working with dense representations of words

### 4. **Positional Embeddings** (`positional_embeddings.ipynb`)
   - Positional encoding techniques
   - Absolute and relative position embeddings
   - Sinusoidal positional embeddings (Transformer approach)
   - Why positional information is crucial for transformers
   - Combining positional and token embeddings

## Key Concepts Covered

### Foundational Concepts
- **Tokenization**: Breaking down text into meaningful units
- **Embeddings**: Converting discrete tokens into continuous vector representations
- **Position Encoding**: Adding positional information to embeddings
- **Vocabulary**: Building and managing token vocabularies

### Techniques & Algorithms
- Regular Expression (regex) based splitting
- Byte Pair Encoding (BPE) - Subword tokenization
- Word2Vec embeddings
- PyTorch tensor operations and neural network layers

### Tools & Libraries
- **PyTorch**: Neural network framework
- **Tiktoken**: OpenAI's fast BPE tokenizer
- **Gensim**: Word embeddings and NLP library
- **Regex**: Pattern matching for text processing

## Learning Path

1. Start with **tokenizing-text.ipynb** to understand basic text tokenization
2. Move to **byte_pair_encoding.ipynb** for advanced tokenization techniques
3. Learn embeddings with **word_vec.ipynb**
4. Explore positional encodings with **positional_embeddings.ipynb**
5. Review **BY_MYSELF/** folder for personal implementations and notes

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or VS Code with Jupyter extension
- Required packages (install via pip):
  ```bash
  pip install torch gensim tiktoken
  ```

### Running the Notebooks
1. Clone the repository
2. Install dependencies
3. Open any notebook in Jupyter or VS Code
4. Run cells sequentially to follow along

## Personal Implementation Notes

The `BY_MYSELF/` folder contains:
- **tokenizing_myself.ipynb**: Custom implementation of tokenization concepts
- Reference materials on Computer Networks and Operating Systems for context

## Next Stages

This repository is structured to progress through multiple stages:
- **Stage 1**: Text Generation & Tokenization (Current)
- **Stage 2**: Transformer Architecture
- **Stage 3**: Training Models
- **Stage 4**: Advanced Techniques & Optimization

## Resources

- **Tokenization**: Understanding different tokenization strategies for NLP
- **Embeddings**: Dense representations capturing semantic information
- **Transformers**: Position-aware self-attention mechanisms
- **Pre-trained Models**: Leveraging existing tokenizers and embeddings

## About This Study

This is an educational repository documenting a comprehensive study of Large Language Models. It combines theoretical understanding with practical implementation, providing a solid foundation for anyone interested in understanding how modern language models work at their core.

---

**Repository**: [GitHub - LLM-From-Scratch](https://github.com/AfzalSurti/LLM-From-Scratch)  
**Last Updated**: January 2026  
**Status**: In Progress - Stage 1 Complete
