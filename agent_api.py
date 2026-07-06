import os
from fastapi import FastAPI, Form
from dotenv import load_dotenv
from rag_module import MedicalRAGEngine
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

app = FastAPI(title="AI Copilot Communications Gateway")
rag_engine = MedicalRAGEngine(api_key=api_key)


@app.post("/sms/incoming")
async def handle_incoming_query(Body: str = Form(...), From: str = Form(...)):
    """
    Receives text queries regarding reports, queries the internal Vector DB,
    and returns a contextual message.
    """
    print(f"Received message from {From}: {Body}")

    # 1. Query Vector store for relevant medical logic
    context = rag_engine._retrieve_medical_context(Body)

    # 2. Run an agent conversation layer using Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.3)

    agent_prompt = (
        f"You are a helpful medical assistant agent answering a user query over a messaging interface.\n"
        f"User Query: {Body}\n\n"
        f"Relevant Guidelines:\n{context}\n\n"
        f"Provide a concise, clear answer suitable for text messaging. Keep it brief and professional."
    )

    response = llm.invoke(agent_prompt)

    # Return response payload structured for webhooks
    return {
        "recipient": From,
        "reply_message": response.content
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)