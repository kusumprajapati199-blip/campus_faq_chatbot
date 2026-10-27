# Campus Service Chatbot with Semantic FAQ Matching

An end-to-end NLP retrieval agent built for CSE276 Continuous Assessment to accurately resolve campus administrative inquiries across Hostel, Library, Fees, ID Cards, Transport, and Examinations.

## 🔗 Live Application Demo
[Live Streamlit Deployment URL](https://campusfaqchatbot-7dpvx69vypq4evbngmsnvs.streamlit.app/)

---

## 📌 Project Overview
Traditional keyword search fails when users ask questions using colloquial language, synonyms, or paraphrasing. This system implements and compares three retrieval paradigms on the identical campus FAQ corpus:
1. **Keyword Overlap (Jaccard Similarity Baseline)**
2. **Vector Space Model (TF-IDF + Cosine Similarity)**
3. **Dense Semantic Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)**

---

## 🏗️ System Architecture
