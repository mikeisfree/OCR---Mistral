import os
import json
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

ocr_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": "https://arxiv.org/pdf/2201.04234"
    },
    include_image_base64=True
)

# Convert the OCR response to a dictionary that can be serialized to JSON
ocr_data = ocr_response.model_dump()

# Save the response as JSON
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(ocr_data, f, indent=4)

print("OCR response saved as output.json")