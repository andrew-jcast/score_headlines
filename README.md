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

  Project Structure

  ├── data/
  │   ├── raw/           <- Place input headline files here
  │   └── processed/
  │       └── local/     <- Sentiment analysis results saved here
  ├── models/
  │   └── svm.joblib     <- Pre-trained sentiment model
  ├── score_headlines.py <- Main CLI script
  └── requirements.txt   <- Python dependencies

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
