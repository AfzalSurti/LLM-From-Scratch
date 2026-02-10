# Self-Attention Mechanism - Step-by-Step Guide

This notebook demonstrates how the **self-attention mechanism** works, which is a fundamental component of transformer models used in modern AI systems like GPT and BERT.

---

## Overview

The notebook walks through building self-attention from scratch using PyTorch, starting with simple word embeddings and progressively showing how attention scores and context vectors are computed.

**Example sentence used:** "Your journey starts with one step"

---

## Step-by-Step Explanation

### **STEP 1: Create Input Embeddings**

```python
import torch

inputs=torch.tensor(
    [
        [0.43,0.15,0.89], #your
        [0.55,0.87,0.66], #journey
        [0.57,0.85,0.64], #starts
        [0.22,0.58,0.33], #with
        [0.77,0.25,0.10], #one
        [0.05,0.80,0.55]  #step
    ]
)
```

**What it does:**
- Creates a 6×3 tensor where each row represents a word embedding
- Each word is represented as a 3-dimensional vector (in real models, this is typically 512 or 768 dimensions)
- These embeddings capture semantic meaning of words in numerical form

---

### **STEP 2: Visualize Word Embeddings in 3D Space**

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

words=["your","journey","starts","with","one","step"]

x_coords=inputs[:,0].numpy()  # Extract x coordinates
y_coords=inputs[:,1].numpy()  # Extract y coordinates
z_coords=inputs[:,2].numpy()  # Extract z coordinates

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')

for x,y,z,word in zip(x_coords,y_coords,z_coords,words):
    ax.scatter(x,y,z)  # Plot the point
    ax.text(x,y,z,word,fontsize=10)  # Add word label

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.title('3D Word Embeddings')
plt.show()
```

**What it does:**
- Plots each word as a point in 3D space
- Shows spatial relationships between word embeddings
- Helps visualize how similar words cluster together

---

### **STEP 3: Visualize Embeddings as Vectors from Origin**

```python
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')

colors=['r','g','b','c','m','y']

for(x,y,z,word,color) in zip(x_coords,y_coords,z_coords,words,colors):
    ax.quiver(0,0,0,x,y,z,color=color,arrow_length_ratio=0.05)  # Draw vector from origin
    ax.text(x,y,z,word,fontsize=10,color=color)

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_zlim([0,1])
plt.title('3D Word Embeddings as Vectors')
plt.show()
```

**What it does:**
- Displays embeddings as arrows (vectors) from the origin point (0,0,0)
- Each word gets a unique color
- Shows the magnitude and direction of each embedding vector

---

### **STEP 4: Compute Attention Scores for a Single Query**

```python
query=inputs[1]  # Use "journey" as the query word

attn_scores_2=torch.empty(inputs.shape[0])
for i,x_i in enumerate(inputs):
    attn_scores_2[i]=torch.dot(x_i,query)  # Dot product between each word and query

print("Attention Scores:", attn_scores_2)
```

**Output:**
```
Attention Scores: tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
```

**What it does:**
- Selects "journey" as the query word
- Computes how relevant each word is to "journey" using dot product
- Higher scores mean higher relevance/similarity
- The second score (1.4950) is "journey" with itself - highest because identical

---

### **STEP 5: Normalize Attention Scores (Simple Division)**

```python
attn_weights_2_tmp=attn_scores_2/attn_scores_2.sum()

print("Attention Weights:", attn_weights_2_tmp)
print("Sum of Attention Weights:", attn_weights_2_tmp.sum())
```

**Output:**
```
Attention Weights: tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
Sum of Attention Weights: tensor(1.0000)
```

**What it does:**
- Normalizes scores by dividing each by the total sum
- Ensures all weights sum to 1.0
- Creates a probability distribution over all words

---

### **STEP 6: Apply Softmax (Naive Implementation)**

```python
def softmax_naive(x): 
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive=softmax_naive(attn_scores_2)

print("Attention weights:",attn_weights_2_naive)
print("Sum:",attn_weights_2_naive.sum())
```

**Output:**
```
Attention weights: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
Sum: tensor(1.)
```

**What it does:**
- Implements softmax function from scratch
- Exponentiates scores to emphasize differences
- Normalizes to create probability distribution
- Softmax gives more weight to higher scores compared to simple division

---

### **STEP 7: Apply Softmax (PyTorch Built-in)**

```python
attn_weights_2=torch.softmax(attn_scores_2,dim=0)
print("Attention Weights:", attn_weights_2)
```

**Output:**
```
Attention Weights: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
```

**What it does:**
- Uses PyTorch's optimized softmax function
- Produces same results as naive implementation
- More numerically stable for large values

---

### **STEP 8: Compute Context Vector for Single Query**

```python
query=inputs[1]  # "journey"

context_vec_2=torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2+=attn_weights_2[i]*x_i  # Weighted sum

print("Context Vector:", context_vec_2)
```

**Output:**
```
Context Vector: tensor([0.4419, 0.6515, 0.5683])
```

**What it does:**
- Creates a new embedding for "journey" that incorporates information from all other words
- Each word contributes proportionally to its attention weight
- Result is a context-aware representation of "journey"
- This is the final output of self-attention for one word

---

### **STEP 9: Compute All Attention Scores (Using Loop)**

```python
attn_scores = torch.empty(inputs.shape[0], inputs.shape[0])

for i, x_i in enumerate(inputs):  # Each input as query
    for j, x_j in enumerate(inputs):  # Each input as key
        attn_scores[i, j] = torch.dot(x_i, x_j)  # Dot product

print(attn_scores)
```

**Output:**
```
tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
        [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
        [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
        [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
        [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
        [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])
```

**What it does:**
- Creates a 6×6 matrix of attention scores
- Row i, column j = how relevant word j is to word i
- Diagonal has highest values (each word is most similar to itself)
- This is the inefficient way using nested loops

---

### **STEP 10: Compute All Attention Scores (Matrix Multiplication)**

```python
attn_scores=inputs @ inputs.T  # Matrix multiplication

print(attn_scores)
```

**Output:**
```
tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
        [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
        [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
        [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
        [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
        [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])
```

**What it does:**
- Computes the same scores as Step 9 but using efficient matrix multiplication
- `inputs @ inputs.T` multiplies the input matrix by its transpose
- Much faster than nested loops, especially for large inputs
- This is how it's done in practice

---

### **STEP 11: Apply Softmax to All Scores**

```python
attn_weights=torch.softmax(attn_scores,dim=-1)

print(attn_weights)
```

**Output:**
```
tensor([[0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452],
        [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581],
        [0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565],
        [0.1435, 0.2074, 0.2046, 0.1462, 0.1263, 0.1720],
        [0.1526, 0.1958, 0.1975, 0.1367, 0.1879, 0.1295],
        [0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]])
```

**What it does:**
- Applies softmax to each row of the attention scores matrix
- `dim=-1` means apply along the last dimension (rows)
- Each row now sums to 1.0, creating probability distributions
- These are the final attention weights for all word pairs

**Important Note:**
- `dim=0` would apply softmax across columns (wrong for self-attention)
- `dim=1` applies softmax across rows (correct)
- `dim=-1` applies softmax across the last dimension (same as `dim=1` for 2D matrices)

---

### **STEP 12: Verify Attention Weights Sum to 1**

```python
row_2_sum=attn_weights[2].sum()

print(row_2_sum)
print(attn_weights[2])
```

**Output:**
```
tensor(1.0000)
tensor([0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565])
```

**What it does:**
- Checks that row 2 (word "starts") attention weights sum to exactly 1.0
- Verifies the softmax worked correctly
- Shows the attention distribution for "starts" across all words

---

### **STEP 13: Compute All Context Vectors**

```python
all_context_vecs=attn_weights @ inputs
print("All Context Vectors:\n", all_context_vecs)
```

**Output:**
```
All Context Vectors:
 tensor([[0.4421, 0.5931, 0.5790],
        [0.4419, 0.6515, 0.5683],
        [0.4431, 0.6496, 0.5671],
        [0.4304, 0.6298, 0.5510],
        [0.4671, 0.5910, 0.5266],
        [0.4177, 0.6503, 0.5645]])
```

**What it does:**
- Computes context vectors for all 6 words simultaneously
- Multiplies the attention weights matrix (6×6) by the input embeddings (6×3)
- Result is 6×3 matrix where each row is a context-aware embedding
- Each word's new representation considers all other words with appropriate weights

---

## Key Concepts Summary

### **What is Self-Attention?**
Self-attention allows each word in a sequence to look at all other words and determine which ones are most relevant to it. The output is a new representation that incorporates contextual information.

### **Main Steps:**
1. **Query, Key, Value**: Each word can act as a query (asking), key (being asked about), or value (contributing information)
2. **Attention Scores**: Compute similarity between query and all keys using dot product
3. **Attention Weights**: Apply softmax to scores to get a probability distribution
4. **Context Vector**: Weighted sum of values using attention weights

### **Why Matrix Multiplication?**
- Computing all attention scores with nested loops is slow: O(n²) operations
- Matrix multiplication computes everything in parallel: much faster on GPUs
- `inputs @ inputs.T` computes all pairwise dot products at once

### **Why Softmax?**
- Simple normalization (division by sum) treats all scores equally
- Softmax exponentiates scores first, which amplifies differences
- Higher scores get proportionally more weight
- Creates a proper probability distribution (all values between 0 and 1, sum to 1)

---

## Mathematical Formula

For self-attention:

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

Where:
- Q = Query matrix
- K = Key matrix  
- V = Value matrix
- d_k = dimension of keys (3 in our example)
- √d_k = scaling factor (not used in this simple example)

In this notebook, Q = K = V = inputs (simplified self-attention)

---

## Real-World Applications

This mechanism is used in:
- **GPT models**: For text generation
- **BERT**: For understanding context
- **Vision Transformers**: For image processing
- **Translation models**: For aligning source and target languages

---

## Next Steps

To build a full transformer:
1. Add Query, Key, Value weight matrices (trainable parameters)
2. Add scaling factor (divide by √d_k)
3. Implement multi-head attention (multiple attention computations in parallel)
4. Add positional encodings
5. Stack multiple attention layers
6. Add feedforward networks

---

## Requirements

```bash
pip install torch matplotlib
```

## Usage

Simply run each cell in order to see the self-attention mechanism in action!