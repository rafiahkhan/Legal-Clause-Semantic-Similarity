# Legal Clause Semantic Similarity – Baseline NLP Models

This repository contains the implementation of **two deep learning baselines** for identifying **semantic similarity between legal clauses**.  
All models were trained **from scratch** using **Keras/TensorFlow** without any pre-trained transformers or fine-tuned legal models.  
This project was developed as part of **CS452 – Deep Learning Assignment 2**.

---

## Project Overview

The goal of this project is to design and train NLP models capable of identifying whether two legal clauses convey similar meaning.  
We compare the performance of two non-transformer baseline architectures:

1. **Siamese BiLSTM**  
2. **BiLSTM + Attention (Fixed)**  

Both architectures learn representations of legal text pairs and classify them as **Similar** or **Dissimilar**.

---

## Dataset

- **Source:** Provided legal clause dataset (`/kaggle/input/legalclausedataset`)
- **Preprocessing:**
  - Lowercased and cleaned textual data
  - Tokenized using Keras `Tokenizer`
  - Padded to a fixed sequence length (`max_len = 120`)
- **Pair Generation:**
  - **Positive pairs (label = 1):** Clauses of the same type  
  - **Negative pairs (label = 0):** Clauses of different types  
  - Dataset balanced between positive and negative examples
- **Dataset Split:**
  | Split | Percentage | Purpose |
  |--------|-------------|----------|
  | Train | 70% | Model training |
  | Validation | 15% | Early stopping and tuning |
  | Test | 15% | Final evaluation |

---

## Model Architectures

### Siamese BiLSTM
- Shared Encoder:  
  `Embedding → BiLSTM(128) → Dense(128)`
- Pair Representation:  
  Concatenate `[u, v, |u−v|, u*v]`
- Classifier:  
  `Dense(256, relu) → Dropout(0.3) → Dense(64, relu) → Dense(1, sigmoid)`
- Trainable Parameters: ~5.9M

### BiLSTM + Attention (Fixed)
- Encoder:  
  `Embedding → BiLSTM(128, return_sequences=True) → AttentionVector`
- Classifier:  
  `Dense(256, relu) → Dropout(0.3) → Dense(64, relu) → Dense(1, sigmoid)`
- Trainable Parameters: ~6.1M

**Training Settings**
- Loss: Binary Crossentropy  
- Optimizer: Adam  
- Batch Size: 64  
- Epochs: 12 (with Early Stopping, patience=3)  
- Embeddings: Randomly initialized (trained from scratch)

---

## Conclusion
Both non-transformer models are strong baselines.  
The Siamese BiLSTM is faster and extremely stable, while the BiLSTM + Attention captures more subtle semantic context.

---

## How to Run the Notebook

1. Open the project in **Kaggle** or **Google Colab**.
2. Upload the dataset: https://www.kaggle.com/datasets/bahushruth/legalclausedataset

