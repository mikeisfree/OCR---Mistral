from flask import Flask, render_template, request, jsonify, Response
import os
import json
import time
from mistralai import Mistral
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
MAX_RETRIES = 3

def process_ocr_with_retry(url, include_image_base64=True):
    """Process OCR with retry logic for reliability"""
    retries = 0
    last_exception = None
    
    while retries < MAX_RETRIES:
        try:
            api_key = os.environ.get("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY environment variable not set")
                
            client = Mistral(api_key=api_key)
            
            logger.debug(f"Sending OCR request for URL: {url}")
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": url
                },
                include_image_base64=include_image_base64
            )
            logger.debug("OCR request successful")
            
            return ocr_response
        except Exception as e:
            last_exception = e
            retries += 1
            logger.error(f"OCR attempt {retries} failed: {str(e)}")
            # Exponential backoff
            time.sleep(2 ** retries)
    
    # If we've exhausted retries, raise the last exception
    logger.error(f"All OCR attempts failed: {str(last_exception)}")
    raise last_exception

# Main UI route
@app.route('/')
def index():
    return render_template('index.html')

# Process OCR from UI
@app.route('/process_ocr', methods=['POST'])
def process_ocr():
    try:
        url = request.form['url']
        output_format = request.form['format']
        
        logger.info(f"Processing OCR for URL: {url}, format: {output_format}")
        ocr_response = process_ocr_with_retry(url)
        
        if output_format == 'json':
            ocr_data = ocr_response.model_dump()
            with open("output.json", "w", encoding="utf-8") as f:
                json.dump(ocr_data, f, indent=4)
            return jsonify({"status": "success", "message": "OCR response saved as output.json"})
        else:
            with open("output.md", "w", encoding="utf-8") as f:
                f.write("# OCR Response\n\n")
                f.write(str(ocr_response))
            return jsonify({"status": "success", "message": "OCR response saved as output.md"})
    except Exception as e:
        logger.error(f"Error in process_ocr: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

# API endpoint for markdown output
@app.route('/api/markdown', methods=['GET'])
def api_markdown():
    try:
        url = request.args.get('url')
        if not url:
            return jsonify({"status": "error", "message": "URL parameter is required"}), 400
        
        logger.info(f"API Markdown: Processing OCR for URL: {url}")
        ocr_response = process_ocr_with_retry(url, include_image_base64=False)
        
        # Extract and combine markdown from all pages
        markdown_content = ""
        for page in ocr_response.pages:
            markdown_content += page.markdown + "\n\n"
        
        logger.debug("Returning markdown response")
        return Response(markdown_content, mimetype='text/markdown')
    except Exception as e:
        logger.error(f"Error in api_markdown: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

# API endpoint for JSON output
@app.route('/api/json', methods=['GET'])
def api_json():
    try:
        url = request.args.get('url')
        if not url:
            return jsonify({"status": "error", "message": "URL parameter is required"}), 400
        
        logger.info(f"API JSON: Processing OCR for URL: {url}")
        ocr_response = process_ocr_with_retry(url)
        
        # Convert to dictionary and ensure it's JSON serializable
        ocr_data = ocr_response.model_dump()
        
        logger.debug("Returning JSON response")
        return jsonify(ocr_data)
    except Exception as e:
        logger.error(f"Error in api_json: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

# Simple test endpoint
@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({"status": "success", "message": "API is working"})

if __name__ == '__main__':
    # Print all registered routes for debugging
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint}: {rule.rule}")
    
    # Run the app
    app.run(debug=True)
