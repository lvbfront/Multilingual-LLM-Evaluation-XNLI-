# 🌍 Multilingual LLM Evaluation (XNLI)

## 📌 Overview
This project investigates whether multilingual language models (MLLMs) perform better in English compared to other languages. It focuses on evaluating different inference strategies across multiple languages using the **XNLI dataset**.

The study explores the performance gap between English and non-English languages and analyzes methods to improve multilingual reasoning.

---

## 🎯 Objectives
- Evaluate multilingual model performance across languages  
- Compare different inference approaches  
- Measure the performance gap between English and non-English tasks  

---

## 🧠 Methods Compared
We evaluate three main approaches:

1. **Direct Inference**  
   Model processes input in the original language  

2. **Machine Translation (MT)**  
   Input is translated into English before inference  

3. **Self-Translation**  
   Model translates input internally before solving the task  

---

## 📊 Dataset
- **XNLI (Cross-lingual Natural Language Inference)**  
- Covers multiple languages for evaluating reasoning and inference  

Languages include:
- High-resource: English, Spanish, French, German  
- Low-resource: Hindi, Swahili, Basque  

---

## ⚙️ Models
- XGLM (multilingual-focused)  
- LLaMA (strong English performance with multilingual capabilities)  

---

## 📈 Evaluation Metrics
- **Accuracy** → Correct predictions for inference tasks  
- **BLEU** → Translation quality  
- **COMET** → Neural evaluation for translation  
- **Cross-lingual Performance Gap** → Difference between English and other languages  

---

## 🔍 Key Findings
- Multilingual models generally perform better in English  
- Translation-based methods improve performance in many cases  
- Self-translation can reduce dependency on external translation systems  
- Performance drops are more significant in low-resource languages  

---

## 💡 Example
**Input (Spanish):**  
"Andy planta 90 geranios y 40 menos petunias que geranios..."

**Translated Input:**  
"Andy plants 90 geraniums and 40 fewer petunias..."

**Output:**  
`140 flowers`

---
## Some Results:
<img width="1343" height="520" alt="image" src="https://github.com/user-attachments/assets/1dac1d0b-5942-47c3-acb2-99eeba88c88c" />

## 🛠️ Tech Stack
- Python  
- Hugging Face Transformers  
- PyTorch  

---

## 🚀 Future Work
- Expand evaluation to more datasets (XCOPA, MGSM, PAWS-X)  
- Improve self-translation techniques  
- Fine-tune models for low-resource languages  

---

## 📎 Project Motivation
Multilingual AI systems are widely used, yet their performance is uneven across languages. This project highlights these gaps and explores practical approaches to improve fairness and accuracy in multilingual AI.

---

## 👨‍💻 Author
Abdullah  
Artificial Intelligence Student  
