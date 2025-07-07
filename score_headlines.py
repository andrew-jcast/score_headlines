#!/usr/bin/env python
"""
Headline Sentiment Analyzer

A simple executable that:
1. Pulls headlines from NYT and/or Chicago Tribune locally
2. Processes them for embeddings
3. Uses an SVM model to produce sentiment scores
"""
import argparse
import datetime
import sys
from pathlib import Path
from typing import List

import numpy as np
import joblib
from sentence_transformers import SentenceTransformer


def generate_embeddings(
    headlines: List[str], model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """Generate embeddings for headlines using sentence transformers."""
    if not headlines:
        return np.array([])

    print(f"Generating embeddings for {len(headlines)} headlines")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(headlines)

    return embeddings


def predict_sentiment(
    embeddings: np.ndarray, model_path: str = "models/svm.joblib"
) -> List[str]:
    """Predict sentiment for embeddings using the trained SVM model."""
    if embeddings.size == 0:
        return []

    print("Predicting sentiment scores")

    # Check if model file exists
    if not Path(model_path).exists():
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    # Load the model
    try:
        clf = joblib.load(model_path)
    except (ValueError, IOError, KeyError) as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Make predictions
    predictions = clf.predict(embeddings)

    return predictions.tolist()


def save_results(
    headlines: List[str],
    sentiments: List[str],
    source: str,
    output_dir: str = "data/processed/local",
):
    """Save headlines and their sentiment predictions to a file."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"headline_scores_{source}_{timestamp}.txt"

    output_path = Path(output_dir) / filename
    with open(output_path, "w", encoding="utf-8") as f:
        for headline, sentiment in zip(headlines, sentiments):
            f.write(f"{sentiment},{headline}\n")

    print(f"Results saved to: {output_path}")
    return str(output_path)


def main():
    """Main function to orchestrate the headline sentiment analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze sentiment of news headlines from a file"
    )
    parser.add_argument(
        "input_file",
        help="Input file name, please place the file in the data/raw directory",
    )
    parser.add_argument("source", help="News source (e.g. nyt or chicagotribune)")
    parser.add_argument(
        "--model-path",
        default="models/svm.joblib",
        help="Path to the trained SVM model (default: models/svm.joblib)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/local",
        help="Directory to save results (default: local directory)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Display results without saving to file"
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        print(
            "Error: Missing required arguments. "
            "Usage: python score_headlines.py <input_file> <source>"
        )
        sys.exit(1)
    # Fetch headlines from the input file
    all_headlines = []
    sources = []

    # Build the full path to the input file in data/raw
    input_path = Path("data/raw") / args.input_file

    # Check if the file exists
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        print(f"Please ensure {args.input_file} is placed in the data/raw directory")
        sys.exit(1)
    # Check file extension
    if not input_path.suffix == ".txt":
        print(f"Error: Expected a .txt file, but got {input_path.suffix}")
        print("Please provide a text file with one headline per line")
        sys.exit(1)

    # Read headlines from the input file
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        all_headlines.extend(lines)
        sources.extend([args.source] * len(lines))

    if not all_headlines:
        print("No headlines found in the input file. Exiting.")
        sys.exit(1)

    # Generate embeddings
    embeddings = generate_embeddings(all_headlines)

    # Predict sentiments
    sentiments = predict_sentiment(embeddings, args.model_path)

    # Display summary
    print("\n" + "=" * 80)
    print("Sentiment Analysis Summary")
    print("=" * 80)

    sentiment_counts = {}
    for sentiment in sentiments:
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

    print(f"Total headlines analyzed: {len(all_headlines)}")

    if not args.no_save:
        save_results(all_headlines, sentiments, args.source, args.output_dir)
    else:
        print("Results will not be saved to file. Use --no-save to skip saving.")
    # Display a few examples
    print("\nSample Results:")
    print("-" * 80)
    for i in range(min(5, len(all_headlines))):
        print(f"[{sentiments[i]}] {all_headlines[i][:100]}")

    print("\nRun completed.")


if __name__ == "__main__":
    main()
