# Stage 1: Text Generation - Tokenization & Word Embeddings

This folder contains foundational concepts for building Large Language Models (LLMs), focusing on text preprocessing, tokenization techniques, and word embeddings.

## 📁 Project Structure

```
text-genration/
├── tokenizing-text.ipynb          # Custom tokenizer implementation
├── byte_pair_encoding.ipynb       # BPE tokenization using tiktoken
├── word_vec.ipynb                 # Word embeddings with PyTorch
├── the-verdict.txt                # Sample text data
└── BY_MYSELF/                     # Custom practice implementations
    ├── tokenizing_myself.ipynb
    ├── Computer_Network.txt
    └── Operating_System.txt
```

## 📚 Learning Modules

### 1. Text Tokenization ([tokenizing-text.ipynb](tokenizing-text.ipynb))

**Objective**: Build a custom tokenizer from scratch to understand the fundamentals of text preprocessing for LLMs.

**Key Concepts Covered**:
- **Text Preprocessing**: Using regex to split text into tokens while preserving punctuation
- **Vocabulary Building**: Creating sorted unique token sets and calculating vocabulary size
- **Token ID Mapping**: Bidirectional mapping between tokens and integer IDs
- **SimpleTokenizerV1**: Basic encode/decode functionality
- **SimpleTokenizerV2**: Enhanced version with special tokens (`<|unk|>`, `<|endoftext|>`)

**What You'll Learn**:
- Split text using regex patterns: `r'([,.:;?_!"()\']|--|\s)'`
- Create vocabulary dictionaries for token-to-ID conversion
- Handle unknown words with `<|unk|>` token
- Mark document boundaries with `<|endoftext|>` token
- Decode token IDs back to readable text

**Code Highlights**:
```python
class SimpleTokenizerV2:
    def encode(self, text):
        # Tokenize, handle unknown tokens, return IDs
    def decode(self, ids):
        # Convert IDs back to text
```

---

### 2. Byte Pair Encoding ([byte_pair_encoding.ipynb](byte_pair_encoding.ipynb))

**Objective**: Implement industry-standard tokenization using OpenAI's tiktoken library (GPT-2 tokenizer).

**Key Concepts Covered**:
- **BPE Algorithm**: Subword tokenization technique used in modern LLMs
- **tiktoken Library**: Production-grade tokenizer from OpenAI
- **Context Windows**: Understanding how LLMs process sequential data
- **Sliding Window Technique**: Creating input-target pairs for training

**What You'll Learn**:
- Use `tiktoken.get_encoding("gpt2")` for state-of-the-art tokenization
- Handle special tokens with `allowed_special` parameter
- Create training examples with context windows
- Generate input-target pairs for next-token prediction
- Understand token-based text length (vs character count)

**Key Implementation**:
```python
context_size = 4
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    # Creates progressive context-target pairs
```

---

### 3. Word Embeddings ([word_vec.ipynb](word_vec.ipynb))

**Objective**: Convert tokens into dense vector representations that capture semantic meaning.

**Key Concepts Covered**:
- **Word2Vec**: Pre-trained Google News embeddings (300 dimensions)
- **PyTorch Embeddings**: Creating custom embedding layers
- **Embedding Dimensions**: Understanding vocab_size and output_dim
- **Reproducibility**: Using `torch.manual_seed()` for consistent results

**What You'll Learn**:
- Load pre-trained embeddings with gensim
- Create PyTorch embedding layers: `torch.nn.Embedding(vocab_size, output_dim)`
- Convert token IDs to dense vectors
- Initialize random embeddings for custom vocabularies
- Scale embeddings for large vocabularies (50,257 tokens for GPT-2)

**Technical Details**:
```python
vocab_size = 50257      # GPT-2 vocabulary
output_dim = 256        # Embedding dimensions
embedding = torch.nn.Embedding(vocab_size, output_dim)
```

---

### 4. Custom Practice ([BY_MYSELF/](BY_MYSELF))

**Personal Implementation**: Applying learned concepts to custom datasets.

**Contents**:
- **tokenizing_myself.ipynb**: Custom tokenizer for technical texts
- **Computer_Network.txt**: Domain-specific corpus
- **Operating_System.txt**: Domain-specific corpus

**Practice Focus**:
- Building separate vocabularies for different domains
- Custom SimpleTokenizerV2 implementations
- Comparing vocabulary sizes across different text types
- Handling technical terminology and special tokens

---

## 🎯 Learning Outcomes

By completing this stage, you will understand:

1. ✅ **Text Preprocessing**: How raw text is converted into machine-readable tokens
2. ✅ **Tokenization Strategies**: Difference between word-level, character-level, and subword (BPE) tokenization
3. ✅ **Vocabulary Management**: Building and managing token vocabularies
4. ✅ **Special Tokens**: Purpose of `<|unk|>`, `<|endoftext|>`, and other control tokens
5. ✅ **Embeddings**: Converting discrete tokens into continuous vector representations
6. ✅ **Context Windows**: How LLMs process sequential information for prediction

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries**:
  - `tiktoken` - OpenAI's BPE tokenizer
  - `torch` - PyTorch for embeddings
  - `gensim` - Pre-trained word vectors
  - `re` - Regular expressions for text processing

---

## 🚀 Getting Started

### Installation
```bash
pip install tiktoken torch gensim
```

### Running the Notebooks
1. Start with [tokenizing-text.ipynb](tokenizing-text.ipynb) to understand basic tokenization
2. Progress to [byte_pair_encoding.ipynb](byte_pair_encoding.ipynb) for advanced BPE
3. Explore [word_vec.ipynb](word_vec.ipynb) for embedding concepts
4. Practice with custom datasets in the `BY_MYSELF` folder

---

## 📖 Key Takeaways

- **Tokenization** is the critical first step in NLP pipelines
- **BPE** balances vocabulary size with representation quality
- **Embeddings** transform discrete symbols into learnable continuous representations
- **Context windows** define how much text an LLM "sees" at once
- Modern LLMs (like GPT) use **subword tokenization** to handle rare words efficiently

---

## 🔗 Next Steps

After mastering tokenization and embeddings:
- Proceed to **Attention Mechanisms** (Self-Attention)
- Learn about **Transformer Architecture**
- Build your first **language model** from scratch

---

## 📝 Notes

- The `the-verdict.txt` file serves as sample training data
- Vocabulary sizes vary: custom (~1,100 tokens) vs GPT-2 (50,257 tokens)
- Embedding dimensions are hyperparameters (typically 256-1024 for production models)

---

**Author's Learning Journey**: This stage represents the foundation of understanding how LLMs process and represent text data internally.