import cv2
import numpy as np
import base64
import google.generativeai as genai  # <-- Make sure this is imported cleanly
from google.generativeai import GenerativeModel
from pydantic import BaseModel, Field
from typing import List


# Define schema for structured output
class Finding(BaseModel):
    class_name: str = Field(
        description="The diagnosed condition or anomaly (e.g., Meningioma, Cardiomegaly, Kidney Stone)")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    box: List[int] = Field(
        description="Bounding box coordinates [xmin, ymin, xmax, ymax] scaled to the image dimensions")
    estimated_size_mm: float = Field(description="Estimated size of the anomaly in millimeters")
    location_tags: str = Field(description="Anatomical location details (e.g., Left Kidney, Frontal Lobe)")


class VisionAnalysisResult(BaseModel):
    findings: List[Finding]


class MedicalVisionEngine:
    def __init__(self):
        print("Production Medical Vision Engine initialized.")

    # UPDATE: Accept api_key here
    def analyze_scan(self, image_bytes, conf_thresh=0.40, iou_thresh=0.45, api_key=None):
        try:
            # Configure the SDK with the verified API token
            if api_key:
                genai.configure(api_key=api_key)

            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return {"success": False, "error": "Invalid image format"}

            height, width, _ = image.shape

            model = GenerativeModel("gemini-2.5-flash")

            prompt = f"Analyze this medical image scan. Identify any visible clinical anomalies, provide their relative bounding boxes normalized to the width ({width}) and height ({height}) of the image, approximate their scale in mm, and name the condition."

            cookie_bytes = base64.b64encode(image_bytes).decode("utf-8")
            image_part = {"mime_type": "image/jpeg", "data": cookie_bytes}

            response = model.generate_content(
                [prompt, image_part],
                generation_config={"response_mime_type": "application/json", "response_schema": VisionAnalysisResult}
            )

            import json
            raw_data = json.loads(response.text)

            final_findings = []
            for item in raw_data.get("findings", []):
                if item["confidence"] >= conf_thresh:
                    final_findings.append(item)

                    xmin, ymin, xmax, ymax = item["box"]
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    label = f"{item['class_name']} ({item['confidence'] * 100:.1f}%)"
                    cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            _, encoded_img = cv2.imencode('.jpg', image)
            return {
                "success": True,
                "findings": final_findings,
                "processed_image": encoded_img.tobytes()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}