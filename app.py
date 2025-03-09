from flask import Flask, render_template, request, jsonify, Response
import os
import json
import time
import threading
from mistralai import Mistral
from functools import wraps
import traceback

import logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Configuration
MAX_RETRIES = 3
TIMEOUT = 60  # seconds

def with_error_handling(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            app.logger.error(f"Error: {str(e)}")
            app.logger.error(traceback.format_exc())
            return jsonify({
                "status": "error",
                "message": str(e),
                "error_type": type(e).__name__
            }), 500
    return decorated_function

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
            
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": url
                },
                include_image_base64=include_image_base64
            )
            
            return ocr_response
        except Exception as e:
            last_exception = e
            retries += 1
            # Exponential backoff
            time.sleep(2 ** retries)
    
    # If we've exhausted retries, raise the last exception
    raise last_exception

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_ocr', methods=['POST'])
@with_error_handling
def process_ocr():
    url = request.form['url']
    output_format = request.form['format']
    
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

@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({"status": "success", "message": "API is working"})

# API endpoints without the decorator to simplify debugging
@app.route('/api/markdown', methods=['GET'])
def api_markdown():
    app.logger.debug("Markdown API endpoint called")
    url = request.args.get('url')
    if not url:
        return jsonify({"status": "error", "message": "URL parameter is required"}), 400
    
    try:
        app.logger.debug(f"Processing OCR for URL: {url}")
        ocr_response = process_ocr_with_retry(url, include_image_base64=False)
        
        # Extract and combine markdown from all pages
        markdown_content = ""
        for page in ocr_response.pages:
            markdown_content += page.markdown + "\n\n"
        
        app.logger.debug("Returning markdown response")
        return Response(markdown_content, mimetype='text/markdown')
    except Exception as e:
        app.logger.error(f"Markdown API error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/json', methods=['GET'])
def api_json():
    app.logger.debug("JSON API endpoint called")
    url = request.args.get('url')
    if not url:
        return jsonify({"status": "error", "message": "URL parameter is required"}), 400
    
    try:
        app.logger.debug(f"Processing OCR for URL: {url}")
        ocr_response = process_ocr_with_retry(url)
        
        # Convert to dictionary and ensure it's JSON serializable
        ocr_data = ocr_response.model_dump()
        
        app.logger.debug("Returning JSON response")
        return jsonify(ocr_data)
    except Exception as e:
        app.logger.error(f"JSON API error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

# Create simple test endpoints to verify routing is working
@app.route('/api/test', methods=['GET'])
def api_test():
    return jsonify({"status": "success", "message": "API test endpoint is working"})

# Simplify the API endpoints to isolate the issue
@app.route('/api/markdown')
def api_markdown():
    app.logger.debug("Markdown API endpoint called")
    return jsonify({"status": "success", "message": "Markdown endpoint reached"})

@app.route('/api/json')
def api_json():
    app.logger.debug("JSON API endpoint called")
    return jsonify({"status": "success", "message": "JSON endpoint reached"})

# Add this at the end to print all registered routes
if __name__ == '__main__':
    # Print all registered routes for debugging
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint}: {rule.rule}")
    
    # Make sure debug is True to see detailed error messages
    app.run(debug=True)