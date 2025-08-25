# Hindi → English Neural Machine Translation (Seq2Seq + Attention)

This project implements a **Neural Machine Translation (NMT)** system from **Hindi to English** using a **Bi-LSTM Encoder–Decoder with Attention** in **PyTorch**.  
It is trained on the [Hindi–English Parallel Corpus](https://www.kaggle.com/datasets/vaibhavkumar11/hindi-english-parallel-corpus).

---

## 📌 Features
- **Sequence-to-Sequence (Seq2Seq) architecture** with **Bi-directional LSTM Encoder** and **Attention-based Decoder**.  
- Supports **teacher forcing**, **dropout regularization**, and **beam search decoding**.  
- Evaluation metrics:
  - **Cross-Entropy Loss**
  - **Token-level Accuracy**
  - **BLEU Score (sentence-level, smoothed)**  
- Preprocessing: case-folding, punctuation/digit removal, sentence length filtering, and start/end tokens.  
- **PyTorch DataLoader + Dataset class** for efficient batching.  
- Saves trained model checkpoints for reuse.  

---

## 📂 Dataset
- Source: [Hindi–English Parallel Corpus](https://www.kaggle.com/datasets/vaibhavkumar11/hindi-english-parallel-corpus)  
- Contains **~500K Hindi–English sentence pairs**.  
- For experiments, we used a subset of **50,000 pairs** (configurable).  
- Sentences longer than **20 words** (either Hindi or English) were discarded to stabilize training.  

---

## 🏗 Model Architecture
### Encoder
- Embedding layer (size = `latent_dim`)  
- **Bi-directional LSTM** (`num_layers=2`)  
- Hidden and cell states are concatenated across directions  

### Decoder
- Embedding layer  
- Attention mechanism over encoder outputs  
- LSTM with input = `[embedded_token ; context_vector]`  
- Fully-connected output layer → vocabulary distribution  
- Dropout regularization  

### Seq2Seq Wrapper
- Trains using **teacher forcing** (`teacher_forcing_ratio=0.5`)  
- Inference supports **beam search decoding** (`beam_width=3`)  

---

## ⚙️ Hyperparameters
| Parameter | Value |
|-----------|-------|
| Latent Dim | 512 |
| LSTM Layers | 2 |
| Dropout | 0.3 |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Epochs | 15 |
| Teacher Forcing Ratio | 0.5 |
| Data Size | 50,000 sentence pairs |

---

## 🚀 Training & Evaluation
### Results (on Test Set after 15 epochs):
- **Test Loss:** `0.5955`  
- **Test Accuracy:** `0.8865`  
- **Test BLEU Score:** `0.3619`  

---

## 📊 Sample Translations
| English Input | Ground Truth (Hindi) | Model Prediction |
|---------------|-----------------------|------------------|
| i am happy | मैं खुश हूँ | मैं खुश हूँ |
| what is your name | तुम्हारा नाम क्या है | तुम्हारा नाम क्या है |
| we are going home | हम घर जा रहे हैं | हम घर जा रहे हैं |

*(Note: Sentences are post-processed to remove `START_` and `_END` tokens.)*  

---

## 🛠 How to Run
1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/hindi-english-nmt.git
   cd hindi-english-nmt
   ```
2. Install dependencies:
   ```bash
   pip install torch numpy pandas scikit-learn tqdm nltk
   ```
   *(If using Kaggle/Colab, most are pre-installed.)*
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/vaibhavkumar11/hindi-english-parallel-corpus) and update `data_path` in the code.  
4. Train the model:
   ```bash
   python train.py
   ```
5. Evaluate & generate translations:
   ```bash
   python evaluate.py
   ```

---

## 📈 Future Improvements
- Use **Transformer-based architectures** (e.g., Transformer Encoder–Decoder, BERT embeddings).  
- Apply **subword tokenization** (Byte-Pair Encoding / SentencePiece) to handle rare words.  
- Train on **full dataset (~500k pairs)** for better generalization.  
- Add **greedy vs beam search comparison** and **length normalization**.  
- Optimize with **mixed precision training** for speed.  

---

## 📜 References
- [Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate (2015)](https://arxiv.org/abs/1409.0473)  
- [Kaggle Dataset: Hindi–English Parallel Corpus](https://www.kaggle.com/datasets/vaibhavkumar11/hindi-english-parallel-corpus)  
- [PyTorch Seq2Seq Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)  

---

✍️ **Author**: Shreyas Shimpi
