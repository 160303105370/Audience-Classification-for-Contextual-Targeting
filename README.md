# 📰 Audience Classification for Contextual Targeting in Ad Inventory

This project implements a **real-time audience classification system** that tags news articles into standardized **IAB content categories** for **contextual ad targeting**.  
By combining **CNN-based text modeling**, **author metadata embeddings**, and **numeric features**, the solution balances accuracy with deployment efficiency.

---

## 📂 Dataset

We use the **[Kaggle News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)**, which contains ~200k news headlines and short descriptions across multiple categories from HuffPost.

- **Fields:** `category`, `headline`, `short_description`, `authors`, `date`, `link`  
- **Preprocessing:**  
  - Normalized category names and mapped them to **IAB taxonomy labels**.  
  - Tokenized text (`headline + short_description`) with a capped vocabulary.  
  - Encoded authors using hashing + embedding.  
  - Derived simple numeric features (publish time, text length, number of authors).

---

## 🧠 Model Architecture

- **Text branch:** Multi-kernel 1D CNN over token embeddings (captures n-gram patterns).  
- **Author branch:** Masked-average embeddings for multiple authors.  
- **Metadata branch:** Dense encoding of numeric features (year, month, weekday, text length, #authors).  
- **Fusion:** Concatenated text + author + metadata → Dense layers → Softmax output.  

---

## 🚀 Key Results

- **Test accuracy:** ~78% across 20 IAB categories.  
- **Strong performance** on major classes (F1 ≈ 0.85 for Politics, Hard News, Sports, Wellness).  
- **Competitive macro-F1** despite class imbalance (minor categories like Divorce, Weddings, Weird News).  

---

## ⚡ Deployment with TensorFlow Lite

- Converted the trained Keras model to **TFLite** with dynamic range quantization.  
- **Latency improvement:**  
  - Keras model (CPU): **116.98 ms/example**  
  - TFLite model (CPU): **1.90 ms/example**  
  - → **98% latency reduction**  
- **Prediction parity:** 99.8% match rate (Keras vs. TFLite).  

This makes the model practical for **real-time contextual ad serving**, where millisecond-level inference matters.

---

## 📊 Business Impact

- Enables **contextual targeting** in ad inventory, improving alignment between ads and content.  
- Boosts **direct deal effectiveness** by ensuring accurate tagging with IAB categories.  
- Deployable at **production scale** with low-latency inference, directly supporting programmatic advertising needs.

---

## 🛠️ Tech Stack

- **Python, TensorFlow/Keras, TensorFlow Lite**  
- **Pandas, NumPy, scikit-learn**  
- **Matplotlib/Seaborn** (EDA & visualization)  

---

## 📑 References

- Dataset: [Kaggle News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) by Rishabh Misra  
- IAB Content Taxonomy: [IAB Tech Lab](https://iabtechlab.com/standards/content-taxonomy/)

---

## ▶️ How to Run

1. Clone this repo  
   ```bash
   git clone https://github.com/your-username/audience-classification.git
   cd audience-classification
