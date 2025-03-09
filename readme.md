```markdown
# PDF OCR API

A Flask-based web application and API that performs OCR on PDF documents using Mistral AI's OCR service.

## Project Structure

 ```
```

d:\AI Creations\OCR ├── test - raw/
│   ├── templates/
│   │   └── index.html
│   ├── app_new.py
│   ├── output.json
│   └── output.md
└── README.md

```plaintext

## Features

- Web interface for PDF OCR processing
- REST API endpoints for JSON and Markdown output
- Support for remote PDF URLs
- Automatic retry mechanism with exponential backoff
- Comprehensive error handling and logging

## Prerequisites

- Python 3.x
- Mistral AI API key
- Flask

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install flask mistralai
 ```
```

3. Set up your Mistral AI API key as an environment variable:
```bash
set MISTRAL_API_KEY=your_api_key_here
 ```

## Usage
### Running Locally
1. Start the Flask application:
```bash
python d:\AI Creations\OCR\test - raw\app_new.py
 ```

2. Access the web interface at: http://127.0.0.1:5000
### API Endpoints
1. JSON Output:
```plaintext
GET /api/json?url=https://example.com/document.pdf
 ```
```

2. Markdown Output:
```plaintext
GET /api/markdown?url=https://example.com/document.pdf
 ```
```

3. Test Endpoint:
```plaintext
GET /test
 ```

## How It Works
1. Web Interface :
   
   - Upload PDF URL
   - Choose output format (JSON/Markdown)
   - Results are saved to local files
2. API Endpoints :
   
   - Direct access to OCR functionality
   - Returns processed text in JSON or Markdown format
   - Includes error handling and retry mechanism
3. OCR Processing :
   
   - Uses Mistral AI's OCR service
   - Supports multiple page documents
   - Handles image extraction and text recognition
## Deployment
### Option 1: Python Anywhere (Recommended)
1. Create a Python Anywhere account
2. Upload your code to Python Anywhere
3. Create a new web app:
   - Choose Flask framework
   - Set your working directory
   - Point to app_new.py
4. Set up your environment variable:
   - Add MISTRAL_API_KEY to your .env file
5. Configure your virtual environment
6. Reload your web app
### Option 2: Heroku
1. Create a Procfile:
```plaintext
web: gunicorn app_new:app
 ```

2. Create requirements.txt:
```bash
pip freeze > requirements.txt
 ```

3. Deploy using Heroku CLI:
```bash
heroku create
git push heroku main
 ```

4. Set environment variables:
```bash
heroku config:set MISTRAL_API_KEY=your_api_key_here
 ```


### Option 3: Railway.app
1. Connect your GitHub repository (include nixpacks.toml file)
2. Create new project from repository
3. Add environment variables
4. Deploy automatically from main branch
## Error Handling
- Automatic retry on failed OCR attempts
- Exponential backoff between retries
- Detailed error logging
- User-friendly error messages
## Limitations
- Requires valid PDF URL
- Processing time depends on PDF size and complexity
- File storage is local (consider cloud storage for production)
- Rate limits based on Mistral AI API plan
## Security Considerations
- API key must be kept secure
- Debug mode should be disabled in production
- Consider adding rate limiting
- Implement input validation for URLs