# Transformers from Scratch using PyTorch

An educational implementation of a **Decoder-only Transformer** architecture built from the ground up. This project demonstrates the fundamental building blocks of modern Large Language Models (LLMs) using PyTorch and PyTorch Lightning for streamlined training.

## 🚀 Overview

This repository contains a step-by-step implementation of the Transformer architecture, specifically focusing on the decoder components. The model is trained on a toy dataset to predict the next token in a sequence, illustrating how attention mechanisms and positional encodings work in practice.

### Key Features
* **Custom Positional Encoding**: Implementation of sine and cosine-based frequency encodings to provide temporal context.
* **Masked Self-Attention**: A fundamental mechanism that allows the model to attend to previous tokens while masking future ones.
* **PyTorch Lightning Integration**: Utilizes `LightningModule` for structured training, making the code scalable and clean.
* **Residual Connections**: Implementation of "Add & Norm" style connections to stabilize gradient flow.

---

## 🏗️ Architecture Breakdown



### 1. Positional Encoding
Since Transformers do not use recurrence (like RNNs), information about the relative or absolute position of the tokens is injected using the following formula:

$$PE(pos, 2i) = \sin(pos / 10000^{2i/d_{model}})$$
$$PE(pos, 2i+1) = \cos(pos / 10000^{2i/d_{model}})$$

### 2. Attention Mechanism
The core of the model calculates **Query (Q)**, **Key (K)**, and **Value (V)** matrices. A causal mask (lower triangular matrix) is used to ensure that the prediction for a specific position can only depend on known outputs at positions less than it.



### 3. Decoder-Only Transformer
The final model combines:
* **Word Embeddings**: Maps token IDs to continuous vectors.
* **Positional Encodings**: Adds sequence context to embeddings.
* **Masked Attention Layer**: Calculates relational scores between tokens.
* **Residual Connections**: Adds the input of the attention layer to its output to prevent vanishing gradients.
* **Fully Connected Layer**: A final linear layer to project results back to the vocabulary size.

---

## 🛠️ Requirements

Ensure you have Python installed, then install the following dependencies:

```bash
pip install torch lightning

## 💻 Usage

### Dataset Preparation
The model uses a custom vocabulary mapped to IDs to handle text data as tensors. 
* **Vocabulary**: `{'what':0, 'is':1, 'CNN':2, 'Great':3, '<EOS>':4}`.
* **Input Mapping**: Prompts are converted into tensors, such as "What is CNN" or "CNN is What".
* **Labeling**: The dataset is structured for next-token prediction, where each input token corresponds to the subsequent token in the sequence.

### Training the Model
The training process is automated using the PyTorch Lightning `Trainer`.
* **Optimizer**: Uses the Adam optimizer with a learning rate of 0.1 for rapid convergence on small datasets.
* **Loss Function**: Employs Cross-Entropy Loss to quantify the difference between predicted tokens and true labels.
* **Execution**: Run the following block to train the model:

```python
# Initialize the trainer for 30 epochs
trainer = L.Trainer(max_epochs=30)
trainer.fit(model, train_dataloaders=dataloader)

---

## 📊 Results

After training for 30 epochs, the model successfully learns the specific sequence patterns provided in the training data.

* **Input Sequence**: "What", "is", "CNN", "<EOS>"
* **Model Output**: The model correctly predicts "Great" followed by "<EOS>".
* **Inference Logic**: The generation loop continues until the `<EOS>` token is predicted or the `max_length` is reached.

---

## 📜 License

This project is open-source and available under the MIT License.
