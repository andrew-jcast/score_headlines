# Headline Sentiment Analyzer

A comprehensive sentiment analysis tool for news headlines with multiple interfaces: command-line, REST API, and web application.

## Overview

This tool processes news headlines and predicts their sentiment (Optimistic, Pessimistic, or Neutral) using a pre-trained SVM model with sentence embeddings. It offers three ways to interact with the sentiment analysis:

1. **Command-line interface** (`score_headlines.py`) - Batch processing from files
2. **REST API** (`score_headlines_api.py`) - Programmatic access via HTTP endpoints  
3. **Web application** (`streamlit_app.py`) - Interactive user interface

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Web Application

Start the Streamlit web interface for interactive sentiment analysis:

```bash
streamlit run streamlit_app.py
```

Features:
- Interactive headline input (paste, type, or upload files)
- Real-time sentiment scoring with caching
- Visual sentiment distribution charts
- Export results to CSV
- Editable headline preview

### 2. Command Line Interface

Batch process headlines from text files:

```bash
python score_headlines.py <input_file> <source>
```

**Example:**
```bash
python score_headlines.py todaysheadlines.txt nyt
```

**Arguments:**
- `input_file`: Text file name (place in `data/raw/` directory)
- `source`: News source identifier (e.g., 'nyt', 'chicagotribune')

**Optional Arguments:**
- `--model-path`: Path to SVM model (default: `models/svm.joblib`)
- `--output-dir`: Results directory (default: `data/processed/local`)
- `--no-save`: Display results without saving

**Input Format:**
- Place files in `data/raw/` directory
- Use `.txt` format with one headline per line
- UTF-8 encoding required

### 3. REST API

Start the API server for programmatic access:

```bash
python score_headlines_api.py
```

The API runs at `http://localhost:8002` with endpoints:

#### `GET /status`
Health check endpoint to verify the API is running.

#### `POST /score_headlines`
Analyzes sentiment for headlines.

**Request:**
```json
{
  "headlines": ["array of headline strings"],
  "return_ids": false
}
```

**Response:**
```json
{
  "labels": ["Optimistic", "Pessimistic", "Neutral"]
}
```

## Output Format

All interfaces return sentiment predictions as:
- **Optimistic**: Positive sentiment headlines
- **Pessimistic**: Negative sentiment headlines  
- **Neutral**: Neutral sentiment headlines

CLI output saves to timestamped files: `headline_scores_<source>_<timestamp>.txt`

## Project Structure

```
├── data/
│   ├── raw/                  <- Input headline files (.txt)
│   └── processed/            <- Analysis results
│       ├── local/            <- CLI output files
│       └── scraped/          <- Web scraped headlines
├── development/              <- Development scripts
│   ├── libraries/            <- Utility modules
│   └── scrape_score_headlines.py
├── models/
│   └── svm.joblib           <- Pre-trained SVM sentiment model
├── notebooks/               <- Jupyter notebooks for development
├── score_headlines.py       <- Command-line interface
├── score_headlines_api.py   <- REST API server
├── streamlit_app.py         <- Web application
└── requirements.txt         <- Python dependencies
```

## Technical Details

- **Model**: Pre-trained SVM with sentence embeddings (all-MiniLM-L6-v2)
- **Sentiment Labels**: Optimistic, Pessimistic, Neutral
- **Caching**: Client-side caching in web app for performance
- **ID Generation**: BLAKE2b hashing for unique headline identifiers
- **Logging**: Request monitoring and error tracking

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
