# 📘 LSTM Next Word Prediction

A clean and beginner-friendly implementation of **Next Word Prediction** using a **Long Short-Term Memory (LSTM)** neural network built with **PyTorch**. This project demonstrates core **Natural Language Processing (NLP)** and **Deep Learning** concepts and serves as a foundational language modeling project.

---

## 📌 Project Overview

Next Word Prediction is a classic NLP task where a model predicts the most likely next word based on a sequence of previous words. In this project:

* A text corpus is used as training data
* Words are tokenized and encoded numerically
* An LSTM-based neural network is trained to learn language patterns
* The trained model predicts the next word for a given input sequence

This project is ideal for understanding how **language models work internally** before moving to advanced architectures like Transformers.

---

## 🧠 Key Concepts (Short & Clear)

### Tokenization

Text is split into individual words (tokens) so that the model can process language in a structured way.

### Vocabulary & Encoding

Each unique word is mapped to a numerical index. Neural networks work with numbers, not raw text.

### Sequence Modeling

The model learns from word sequences using a sliding window approach:

* Input: previous *N* words
* Output: next word

This teaches the model contextual relationships between words.

### LSTM (Long Short-Term Memory)

LSTM is a type of Recurrent Neural Network (RNN) designed to:

* Capture long-term dependencies
* Retain important contextual information
* Avoid vanishing gradient problems

It is well-suited for language-based tasks.

### Word Embeddings

An embedding layer converts word indices into dense vectors that capture semantic meaning, allowing the model to understand similarity between words.

---

## ⚙️ Model Architecture

* **Embedding Layer** – Converts words into dense vectors
* **LSTM Layer** – Learns sequential patterns in text
* **Fully Connected Layer** – Predicts the next word
* **Loss Function** – Cross-Entropy Loss
* **Optimizer** – Adam

---

## 🛠️ Tech Stack

* Python
* PyTorch
* NLTK
* NumPy
* Jupyter Notebook

---

## 📂 Project Workflow

1. Import and install dependencies
2. Load and preprocess text data
3. Tokenize text and build vocabulary
4. Generate input–target sequences
5. Define the LSTM model
6. Train the model on text data
7. Predict the next word for new inputs

---

## ▶️ Usage

1. Clone the repository

```bash
git clone https://github.com/your-username/lstm-next-word-prediction.git
```

2. Install dependencies

```bash
pip install torch nltk numpy
```

3. Run the notebook

```bash
jupyter notebook LSTM_next_word.ipynb
```

4. Enter a sentence and observe the predicted next word

---

## 🚀 Applications

* Text autocompletion
* Chatbots
* Language modeling
* Sentence generation

---

## 🔮 Future Improvements

* Train on larger datasets
* Add temperature-based sampling
* Generate full sentences
* Compare performance with Transformer models

---

## 👤 Author

**Sambit Ganguly**
Aspiring AI / Machine Learning Engineer

---

⭐ If you find this project useful, consider starring the repository!
