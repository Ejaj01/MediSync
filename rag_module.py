from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import chromadb  # <-- IMPORT THIS HERE

class MedicalRAGEngine:
    def __init__(self, api_key=None):
        # Embeddings belong here in the initialization layer
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=api_key
        )
        
        # FIX: Clear Chroma's system cache to prevent the Streamlit rerun crash
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        
        self.db = Chroma(persist_directory="./chroma_db", embedding_function=self.embeddings)
        print("Medical RAG Engine initialized against production ChromaDB storage.")

    def _retrieve_medical_context(self, finding_name):
        # Dynamic semantic search query replaces dictionary matching
        docs = self.db.similarity_search(finding_name, k=2)
        if docs:
            return "\n\n".join([doc.page_content for doc in docs])
        return "General clinical safety protocols apply. No specific guidelines found."


    def generate_doctor_report(self, finding, size_mm, location, api_key=None):
        reference_context = self._retrieve_medical_context(finding)

        system_instruction = (
            "You are an advanced, clinical-grade AI Medical Copilot specializing in radiology and diagnostic evaluation.\n"
            "Your task is to draft an objective, professional Medical Report based ONLY on the provided Machine Learning "
            "Vision Findings and the attached trusted Open-Source Medical Reference Context.\n\n"
            "CRITICAL RULES:\n"
            "1. Do not assume or extrapolate anything outside the provided reference text.\n"
            "2. Maintain a highly professional, clinical, and reassuring tone.\n"
            "3. If findings indicate a severe risk threshold based on size or location guidelines, flag it clearly under 'CRITICALITY & NEXT STEPS'.\n"
            "4. Always include a legal disclaimer stating this must be reviewed by a human physician."
        )

        user_prompt_template = (
            "### VISION ENGINE DETECTIONS:\n"
            "- Diagnosed Finding: {finding}\n"
            "- Measured Size: {size_mm} mm\n"
            "- Anatomical Location: {location}\n\n"
            "### TRUSTED MEDICAL REFERENCE CONTEXT:\n"
            "{context}\n\n"
            "Please compile the final Clinical Evaluation Report structured with clear sections: "
            "1. Diagnostic Summary, 2. Correlated Symptoms, 3. Recommended Clinical Interventions, 4. Criticality & Next Steps."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("user", user_prompt_template)
        ])

        formatted_prompt = prompt.format(
            finding=finding,
            size_mm=size_mm,
            location=location,
            context=reference_context
        )

        if api_key and api_key.strip() != "":
            try:
                # FIX: Keep this as your text conversational generation engine
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=api_key,
                    temperature=0.1
                )
                response = llm.invoke(formatted_prompt)
                return response.content
            except Exception as e:
                return f"Error executing live Cloud Gemini LLM: {str(e)}"
        else:
            return self._generate_local_deterministic_report(finding, size_mm, location, reference_context)

    def _generate_local_deterministic_report(self, finding, size_mm, location, context):
        is_critical = size_mm > 7.0
        severity_status = "🔴 CRITICAL (Intervention Likely Required)" if is_critical else "🟡 STABLE / MONITORING"

        return (
            f"## CLINICAL EVALUATION REPORT (LOCAL EMULATION MODE)\n\n"
            f"### 1. Diagnostic Summary\n"
            f"- **Observed Pathology:** {finding}\n"
            f"- **Anatomical Mapping:** {location}\n"
            f"- **Calculated Dimension:** {size_mm} mm\n\n"
            f"### 2. Correlated Symptoms\n"
            f"Based on open-source urology data, patient may present with unilateral flank pain radiating to the groin.\n\n"
            f"### 3. Recommended Clinical Interventions\n"
            f"- **Metric Evaluation:** The stone size measures {size_mm}mm. According to reference guidelines, stones exceeding 7mm exhibit a lower spontaneous passage profile (<20%).\n\n"
            f"### 4. Criticality & Next Steps\n"
            f"- **Status Assessment:** {severity_status}\n\n"
            f"*Disclaimer: This automated extraction tool must be verified by a board-certified physician.*"
        )
