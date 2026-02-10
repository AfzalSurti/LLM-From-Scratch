# LLM From Scratch - Study Repository

A comprehensive study repository documenting the journey of building Large Language Models (LLMs) from scratch, alongside foundational deep learning concepts. This project explores neural networks, attention mechanisms, and modern NLP techniques required to understand and implement state-of-the-art models like GPT and Transformers.

## 🎯 Project Overview

This repository contains structured learning materials and hands-on code implementations covering three major learning tracks:

1. **Stage-1: Text Generation & Tokenization** - NLP fundamentals for LLMs
2. **CNN: Image Classification** - Convolutional Neural Networks on CIFAR-10
3. **Attention: Self-Attention Mechanisms** - Core component of Transformer models

Each track includes Jupyter notebooks with detailed explanations, code implementations, and comprehensive README documentation.

---

## 📁 Repository Structure

```
LLMs-from-scratch-study/
├── stage-1/
│   └── text-genration/
│       ├── tokenizing-text.ipynb           # Custom tokenizer implementation
│       ├── byte_pair_encoding.ipynb        # BPE with tiktoken (GPT-2)
│       ├── word_vec.ipynb                  # PyTorch embeddings & Word2Vec
│       ├── the-verdict.txt                 # Sample training data
│       ├── README.md                       # Detailed stage-1 guide
│       └── BY_MYSELF/                      # Personal implementations
│           ├── tokenizing_myself.ipynb     # Custom tokenization practice
│           ├── Computer_Network.txt        # Domain-specific corpus
│           └── Operating_System.txt        # Domain-specific corpus
│
├── CNN/
│   ├── cnn.ipynb                           # Complete CIFAR-10 CNN pipeline
│   ├── best_cnn_model.h5                   # Trained model (H5 format)
│   ├── cnn_final_model.keras               # Trained model (Keras format)
│   ├── README.md                           # CNN theory & usage guide
│   └── predicted_images/                   # Custom image predictions folder
│
├── Attention/
│   ├── slef_attention.ipynb                # Self-attention from scratch
│   ├── RNN.txt                             # RNN reference notes
│   └── README.md                           # Step-by-step attention guide
│
├── DataPreProcessing/
│   └── Tokenization/                       # Additional preprocessing experiments
│
└── README.md                               # This file
```

---

---

## 📚 Learning Tracks

### 🔹 Track 1: Stage-1 - Text Generation & Tokenization

**Focus**: Foundation of NLP for Large Language Models

**What You'll Learn**:
- Build custom tokenizers from scratch (regex-based splitting, vocabulary creation)
- Implement Byte Pair Encoding (BPE) using OpenAI's tiktoken library
- Create word embeddings with PyTorch and understand Word2Vec
- Handle special tokens (`<|unk|>`, `<|endoftext|>`)
- Create context windows for next-token prediction
- Scale to production-level vocabularies (50,257 tokens for GPT-2)

**Key Notebooks**:
1. **[tokenizing-text.ipynb](stage-1/text-genration/tokenizing-text.ipynb)** - SimpleTokenizerV1 & V2 implementations
2. **[byte_pair_encoding.ipynb](stage-1/text-genration/byte_pair_encoding.ipynb)** - Subword tokenization with tiktoken
3. **[word_vec.ipynb](stage-1/text-genration/word_vec.ipynb)** - Embedding layers and vector representations

**Technologies**: `tiktoken`, `torch`, `gensim`, `regex`

👉 **[Full Stage-1 Documentation →](stage-1/text-genration/README.md)**

---

### 🔹 Track 2: CNN - Image Classification with CIFAR-10

**Focus**: Convolutional Neural Networks for Computer Vision

**What You'll Learn**:
- Complete end-to-end CNN pipeline for image classification
- Train models on CIFAR-10 dataset (60,000 32×32 color images, 10 classes)
- Implement convolution, pooling, and dense layers
- Use Early Stopping to prevent overfitting
- Visualize training metrics (accuracy, loss curves)
- Make predictions on custom images
- Save and load trained models (H5 and Keras formats)

**Key Features**:
- **Single-notebook workflow** - Everything in [cnn.ipynb](CNN/cnn.ipynb)
- **Pre-trained models** - Ready-to-use saved models included
- **Custom predictions** - Test on your own images in `predicted_images/`
- **Theory revision** - Comprehensive CNN concepts explained

**CNN Concepts Covered**:
- Convolution & Kernels (Filters)
- Stride & Padding
- Pooling (MaxPooling, AvgPooling)
- Feature Maps & Channels
- Flatten & Dense Layers
- Early Stopping Strategy

**Technologies**: `TensorFlow/Keras`, `matplotlib`, `numpy`

👉 **[Full CNN Documentation →](CNN/README.md)**

---

### 🔹 Track 3: Attention - Self-Attention Mechanisms

**Focus**: Understanding the core of Transformer models

**What You'll Learn**:
- Build self-attention from scratch using PyTorch
- Visualize word embeddings in 3D space
- Compute Query (Q), Key (K), and Value (V) projections
- Calculate attention scores using scaled dot-product attention
- Generate context-aware embeddings for each word
- Understand why attention is crucial for GPT, BERT, and Transformers

**Step-by-Step Process**:
1. Create input embeddings (word vectors)
2. Visualize embeddings as 3D vectors
3. Initialize weight matrices (W_q, W_k, W_v)
4. Compute Q, K, V matrices
5. Calculate attention scores (Q @ K^T)
6. Apply softmax normalization
7. Compute weighted context vectors

**Example Sentence**: *"Your journey starts with one step"*

**Key Insights**:
- Each word "attends" to all other words in the sequence
- Attention scores determine which words are most relevant
- Self-attention captures relationships between words regardless of distance

**Technologies**: `PyTorch`, `matplotlib`, `numpy`

👉 **[Full Attention Documentation →](Attention/README.md)**

---

---

## 🎓 Key Concepts Covered

### Natural Language Processing (NLP)
- **Tokenization**: Breaking text into meaningful units (words, subwords, characters)
- **Byte Pair Encoding (BPE)**: Efficient subword tokenization used in GPT models
- **Word Embeddings**: Dense vector representations capturing semantic meaning
- **Vocabulary Management**: Building and scaling token vocabularies
- **Special Tokens**: Control tokens for marking boundaries and handling unknowns
- **Context Windows**: Sequential data processing for language modeling

### Computer Vision (CV)
- **Convolutional Neural Networks**: Feature extraction from images
- **Convolution Operations**: Pattern detection using kernels/filters
- **Pooling Layers**: Downsampling and feature selection
- **Image Classification**: Multi-class prediction on CIFAR-10 dataset
- **Transfer Learning**: Using pre-trained weights
- **Regularization**: Early stopping to prevent overfitting

### Attention Mechanisms
- **Self-Attention**: Computing relationships between all sequence elements
- **Query-Key-Value (QKV)**: Attention computation paradigm
- **Attention Scores**: Weighted importance of different words
- **Scaled Dot-Product Attention**: Efficient attention calculation
- **Context Vectors**: Enhanced representations using attention
- **Transformer Foundations**: Building blocks of modern LLMs

### Deep Learning Fundamentals
- **PyTorch**: Tensor operations and neural network construction
- **TensorFlow/Keras**: High-level model building and training
- **Optimization**: Adam optimizer, learning rate tuning
- **Training Loops**: Forward pass, loss calculation, backpropagation
- **Evaluation Metrics**: Accuracy, loss curves, confusion matrices
- **Model Persistence**: Saving and loading trained models

---

## 🛠️ Technologies & Libraries

| Category | Tools |
|----------|-------|
| **Deep Learning Frameworks** | PyTorch, TensorFlow, Keras |
| **NLP Libraries** | tiktoken, gensim |
| **Data Processing** | NumPy, Pandas, Regex |
| **Visualization** | Matplotlib, Seaborn |
| **Development** | Jupyter Notebook, VS Code |
| **Version Control** | Git, GitHub |

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.8+**
- **Jupyter Notebook** or **VS Code** with Jupyter extension
- **GPU (Optional)**: Recommended for CNN training

### Installation

1. **Clone the repository**:
```powershell
git clone https://github.com/yourusername/LLMs-from-scratch-study.git
cd LLMs-from-scratch-study
```

2. **Create a virtual environment** (recommended):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# OR
source .venv/bin/activate      # macOS/Linux
```

3. **Install dependencies**:

For **Stage-1 (Text Generation)**:
```powershell
pip install torch tiktoken gensim
```

For **CNN (Image Classification)**:
```powershell
pip install tensorflow matplotlib numpy jupyter
```

For **Attention Mechanisms**:
```powershell
pip install torch matplotlib numpy
```

Or install everything at once:
```powershell
pip install torch tensorflow tiktoken gensim matplotlib numpy jupyter
```

### Running the Notebooks

**Option 1: Jupyter Notebook**
```powershell
jupyter notebook
```
Navigate to the desired folder and open any `.ipynb` file.

**Option 2: VS Code**
1. Open the workspace in VS Code
2. Install the Jupyter extension
3. Open any notebook file
4. Select your Python kernel
5. Run cells sequentially

---

---

## 📖 Recommended Learning Path

### For Complete Beginners
1. **Start with CNN** - Understand basic neural networks and image classification
2. **Move to Stage-1** - Learn NLP fundamentals and tokenization
3. **Finish with Attention** - Master the core mechanism of Transformers

### For NLP-Focused Learners
1. **Stage-1** - Tokenization and embeddings
2. **Attention** - Self-attention mechanisms
3. **CNN** (Optional) - Broaden understanding with computer vision

### For Building LLMs
1. **Stage-1: Text Generation** - Master tokenization with BPE
   - Complete all three notebooks sequentially
   - Practice with custom datasets in `BY_MYSELF/`
2. **Attention: Self-Attention** - Understand attention mechanisms
   - Visualize embeddings and attention scores
   - Grasp Query-Key-Value paradigm
3. **Next**: Implement full Transformer architecture (coming soon)

---

## 🎯 Learning Outcomes

After completing this repository, you will be able to:

✅ **Tokenize text** using custom and production tokenizers (BPE)  
✅ **Create word embeddings** in PyTorch with proper vocabulary management  
✅ **Build CNNs** for image classification from scratch  
✅ **Train and evaluate** deep learning models with early stopping  
✅ **Implement self-attention** and understand attention scores  
✅ **Visualize** embeddings, attention weights, and training metrics  
✅ **Prepare data** for training Large Language Models  
✅ **Understand** the foundations of GPT, BERT, and Transformer models  

---

## 📂 Project Highlights

### ✨ Stage-1 Features
- Custom tokenizer implementations (`SimpleTokenizerV1` & `V2`)
- GPT-2 tokenizer integration with tiktoken
- Word2Vec and PyTorch embedding layers
- Training data preparation with context windows
- Special token handling for production scenarios

### ✨ CNN Features
- Complete CIFAR-10 classification pipeline
- Pre-trained models ready for inference
- Custom image prediction capability
- Comprehensive theory revision notes
- Early stopping implementation

### ✨ Attention Features
- Self-attention built from scratch
- 3D embedding visualizations
- Step-by-step attention score calculation
- Context vector generation
- Mathematical foundations explained

---

## 📚 Additional Resources

### Official Documentation
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [tiktoken GitHub](https://github.com/openai/tiktoken)
- [Gensim Documentation](https://radimrehurek.com/gensim/)

### Recommended Reading
- **"Attention Is All You Need"** - Original Transformer paper
- **"Build a Large Language Model (From Scratch)"** - Sebastian Raschka
- **Deep Learning Specialization** - Andrew Ng (Coursera)
- **Hugging Face NLP Course** - Free comprehensive guide

### Related Repositories
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

---

## 🔄 Repository Status

| Track | Status | Notebooks | Documentation |
|-------|--------|-----------|---------------|
| **Stage-1: Text Generation** | ✅ Complete | 3 main + 1 practice | ✅ Comprehensive |
| **CNN: Image Classification** | ✅ Complete | 1 end-to-end | ✅ Comprehensive |
| **Attention Mechanisms** | ✅ Complete | 1 detailed | ✅ Comprehensive |
| **DataPreProcessing** | 🚧 In Progress | - | - |
| **Transformer Architecture** | 📋 Planned | - | - |
| **Model Training** | 📋 Planned | - | - |

---

## 🤝 Contributing

This is a personal learning repository, but suggestions and improvements are welcome!

- Open an issue for bugs or questions
- Submit pull requests for corrections
- Share your own implementations and experiments

---

## 📝 Notes

- All notebooks are designed to run independently
- Each folder has its own detailed README with step-by-step guides
- Code is heavily commented for educational purposes
- Models and data files are included where applicable

---

## 📧 Contact & Links

**Repository**: [GitHub - LLMs-from-scratch-study](https://github.com/yourusername/LLMs-from-scratch-study)  
**Last Updated**: February 2026  
**Status**: Active Development  

---

## 📜 License

This project is for educational purposes. Feel free to use the code and adapt it for your own learning journey.

---

## 🌟 Acknowledgments

- OpenAI for tiktoken and GPT research
- Sebastian Raschka for LLM learning resources
- TensorFlow and PyTorch communities
- CIFAR-10 dataset creators
- The broader AI/ML research community

---

**Happy Learning! 🚀**

*Building understanding one token, one convolution, and one attention head at a time.*
