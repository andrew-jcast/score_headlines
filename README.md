deploy_headlines
==============================

  # Headline Sentiment Analyzer

  A command-line tool for analyzing sentiment of news headlines using machine learning.

  ## Overview

  This tool processes news headlines from text files and predicts their sentiment (positive, negative, or neutral) using a
  pre-trained SVM model with sentence embeddings.

  ## Installation

  ```bash
  pip install -r requirements.txt
  ```

  Usage

  Basic usage:
  `python score_headlines.py <input_file>, <source>`

  Example:
  `python score_headlines.py todaysheadlines.txt nyt`

  Arguments

  - input_file: Name of the text file containing headlines (must be placed in data/raw/ directory)
  - source: News source identifier (e.g., 'nyt', 'chicagotribune')

  Optional Arguments

  - --model-path: Path to the trained SVM model (default: models/svm.joblib)
  - --output-dir: Directory to save results (default: data/processed/local)
  - --no-save: Display results without saving to file

  Input Format

  Place your headline files in the data/raw/ directory. Files must:
  - Be in .txt format
  - Contain one headline per line
  - Use UTF-8 encoding

  Example file structure:
  data/raw/todaysheadlines.txt

  Output

  The tool generates:
  - Sentiment predictions for each headline
  - Summary statistics showing sentiment distribution
  - Timestamped output file: headline_scores_<source>_<timestamp>.txt

  Output format (CSV-style):
  positive,Markets surge on strong earnings reports
  negative,Breaking: Major earthquake strikes coastal region
  neutral,City council meets to discuss budget proposal

  # Headline Sentiment Analyzer - API

  ## Overview

  The **Headline Sentiment Analyzer** provides a RESTful API for real-time sentiment analysis of headlines. Built with **FastAPI**, it supports batch processing and optional unique ID generation for each headline.

  ---

  ## Starting the API

  ```
  python score_headlines_api.py
  ```

  The API will start at [http://localhost:8001](http://localhost:8001) by default.

  ---

  ## Endpoints

  ### `GET /status`

  Health check endpoint to verify the API is running.

  **Response:**
  ```json
  {
    "status": "OK"
  }
  ```

  ---

  ### `POST /score_headlines`

  Analyzes sentiment for one or more headlines.

  **Request Body:**
  ```json
  {
    "headlines": ["array of headline strings"],
    "return_ids": false
  }
  ```

  **Parameters:**
  - `headlines` (required): List of headline strings to analyze  
  - `return_ids` (optional): If true, returns unique IDs for each headline

  **Response (with `return_ids=false`):**
  ```json
  {
    "labels": ["Optimistic", "Pessimistic", "Neutral"]
  }
  ```

  **Response (with `return_ids=true`):**
  ```json
  {
    "results": [
      {
        "id": "eac3c48b39894338157b",
        "label": "Optimistic"
      },
      {
        "id": "182d3ee7fdae04e43bd2",
        "label": "Pessimistic"
      }
    ]
  }
  ```

  ---

  ## Usage Examples

  ### Python (with `requests`)

  ```python
  import requests

  # Score headlines without IDs
  response = requests.post(
      "http://localhost:8001/score_headlines",
      json={
          "headlines": [
              "Markets surge on strong earnings",
              "Economic uncertainty looms ahead"
          ]
      }
  )
  print(response.json())
  # Output: {"labels": ["Optimistic", "Pessimistic"]}

  # Score headlines with unique IDs
  response = requests.post(
      "http://localhost:8001/score_headlines",
      json={
          "headlines": ["Tech stocks rally"],
          "return_ids": True
      }
  )
  print(response.json())
  # Output: {"results": [{"id": "a1b2c3d4e5", "label": "Optimistic"}]}
  ```

  ---

  ### cURL

  ```bash
  # Check API status
  curl http://localhost:8001/status

  # Score headlines
  curl -X POST http://localhost:8001/score_headlines \
    -H "Content-Type: application/json" \
    -d '{"headlines": ["Breaking news: Major discovery announced"]}'
  ```

  ---

  ## Error Handling

  The API returns appropriate HTTP status codes:
  - **400 Bad Request**: Empty headline list  
  - **500 Internal Server Error**: Model prediction failure

  ---

  ## Technical Details

  - Uses pre-trained SVM model with sentence embeddings (`all-MiniLM-L6-v2`)  
  - Headline IDs are generated using **BLAKE2b** hash (when requested)  
  - Includes request logging for monitoring  
  - Model loading is **cached** for performance

  Project directory overview:
```
├── data/
│   ├── raw/                <- Place input headline files here
│   └── processed/
│       └── local/         <- Sentiment analysis results saved here
├── models/
│   └── svm.joblib         <- Pre-trained sentiment model
├── score_headlines.py     <- Main CLI script
├── score_headlines_api.py <- Main API hosting script
└── requirements.txt       <- Python dependencies
```


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
