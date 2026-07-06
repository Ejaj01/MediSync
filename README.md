# AI Medical Imaging & Clinical Copilot 🩺🤖

A production-grade, Multimodal Retrieval-Augmented Generation (RAG) agent that bridges computer vision analysis with medical domain guidelines. The system identifies abnormalities in medical scans and cross-references findings with official clinical literature to provide contextual evaluation reports.

🔗 **Live Application:** [Launch Streamlit Web App](https://medical-ai-copilot-ejajfirstragaikh5ojjdhqxzjv3vnxlgxvl.streamlit.app/)

---

## 🚀 Key Architectural Features
* **Multimodal Core:** Seamlessly processes raw medical pixel data alongside complex textual data streams.
* **Computer Vision Engine:** Detects and localizes regions of interest (ROI) via automated bounding boxes.
* **Vector Semantic Search:** Embeds and stores official medical guidelines (e.g., EAU Guidelines on Neuro-Urology) into an isolated `ChromaDB` vector database.
* **Context-Driven Orchestration:** Uses `LangChain` to dynamically query the vector storage and orchestrate LLM responses with `Gemini-1.5`.

---

## 🛠️ Tech Stack & Dependencies
* **Frontend Interface:** Streamlit Cloud
* **LLM & Multi-Modal Processing:** Google Gemini API (`google-generativeai`, `langchain-google-genai`)
* **Vector Database:** ChromaDB (`langchain-chroma`)
* **Language Model Orchestration:** LangChain Core & Community
* **Image Processing:** OpenCV (`opencv-python-headless`) & NumPy
* **Environment Management:** Python 3.10 + `python-dotenv` + `pysqlite3-binary`

---

## 💻 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Ejaj01/medical-ai-copilot.git](https://github.com/Ejaj01/medical-ai-copilot.git)
   cd medical-ai-copilot