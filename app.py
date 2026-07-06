try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # On Windows, pysqlite3 won't be found, so it falls back to native sqlite3 safely
    pass
import streamlit as st
import os
from dotenv import load_dotenv
from vision_module import MedicalVisionEngine
from rag_module import MedicalRAGEngine

# 1. Load the secret environment variables from your .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 2. Initialize our specialized engines
vision_engine = MedicalVisionEngine()
rag_engine = MedicalRAGEngine(api_key=api_key)

st.set_page_config(
    page_title="Medical AI Copilot",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI Medical Imaging & Clinical Copilot")
st.markdown("---")

# Sidebar UI Controls
with st.sidebar:
    st.header("⚙️ Detection Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.40, 0.05)
    iou_threshold = st.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.45, 0.05)
    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** Results are for educational purposes only.")

# Main Layout splitting the screen
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Upload Medical Imagery")
    uploaded_file = st.file_uploader("Upload X-Ray, CT, or Ultrasound scan...",
                                     type=["jpg", "png", "jpeg", "bmp", "tif"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        vision_result = vision_engine.analyze_scan(image_bytes, confidence_threshold, iou_threshold, api_key=api_key)

        if vision_result["success"]:
            st.image(vision_result["processed_image"], caption="Processed Scan (AI Finding Layer)",
                     use_container_width=True)
            st.success("Image processed by Computer Vision Engine.")

            # --- NEW INTERACTIVE COMMUNICATION AI COMPONENT ---
            st.markdown("---")
            st.subheader("💬 AI Copilot Clinical Chat")
            st.caption("Ask specific contextual follow-up questions regarding the report findings below:")

            # Initialize thread state variables
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # Display clear back-and-forth log
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Input field for user query
            if user_query := st.chat_input(
                    "Ex: What are the primary indicators for surgical intervention for this size?"):
                with st.chat_message("user"):
                    st.markdown(user_query)
                st.session_state.chat_history.append({"role": "user", "content": user_query})

                # Grab relevant background reference text via RAG
                current_finding = vision_result["findings"][0]["class_name"] if vision_result["findings"] else "General"
                kb_context = rag_engine._retrieve_medical_context(current_finding)

                # Initialize LLM wrapper execution layer
                from langchain_google_genai import ChatGoogleGenerativeAI

                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.2)

                chat_prompt = (
                    f"You are a consultative medical specialist assistant reviewing a patient case file.\n"
                    f"Context Guidelines:\n{kb_context}\n\n"
                    f"Answer the user query accurately based on these guidelines: {user_query}"
                )

                with st.chat_message("assistant"):
                    response = llm.invoke(chat_prompt)
                    st.markdown(response.content)

                st.session_state.chat_history.append({"role": "assistant", "content": response.content})
            # --------------------------------------------------
        else:
            st.error(f"Vision Processing Error: {vision_result['error']}")
    else:
        st.info("Please upload an image file to begin analysis.")

with col2:
    st.subheader("📋 Clinical Evaluation Report")

    if uploaded_file is not None and vision_result["success"]:
        # Check if we have detected any findings
        if vision_result["findings"]:
            # Extract the first finding detected by our vision engine
            primary_finding = vision_result["findings"][0]

            st.info("🔄 Running Cloud RAG Medical Evaluation...")

            # Call our RAG Brain and pass the key we loaded from our secret file
            doctor_report = rag_engine.generate_doctor_report(
                finding=primary_finding["class_name"],
                size_mm=primary_finding["estimated_size_mm"],
                location=primary_finding["location_tags"],
                api_key=api_key
            )

            # Print the markdown report on screen beautifully
            st.markdown(doctor_report)
        else:
            st.success("✅ Analysis Complete: No clear anomalies detected within current threshold settings.")
    else:
        st.write("The generated clinical analysis will appear here after an image is uploaded.")