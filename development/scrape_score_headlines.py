#!/usr/bin/env python
"""
Headline Sentiment Analyzer

A simple executable that:
1. Pulls headlines from NYT and/or Chicago Tribune
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
from libraries.fetch_headlines import fetch_nyt_headlines, fetch_chicago_tribune_headlines


def generate_embeddings(headlines: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Generate embeddings for headlines using sentence transformers."""
    if not headlines:
        return np.array([])

    print(f"Generating embeddings for {len(headlines)} headlines...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(headlines)

    return embeddings


def predict_sentiment(embeddings: np.ndarray, model_path: str = "models/svm.joblib") -> List[str]:
    """Predict sentiment for embeddings using the trained SVM model."""
    if embeddings.size == 0:
        return []

    print("Predicting sentiment scores...")

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


def save_results(headlines: List[str], sentiments: List[str], source: str,
                 output_dir: str = "data/processed/scraped"):
    """Save headlines and their sentiment predictions to a file."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{output_dir}/sentiment_analysis_{source}_{timestamp}.txt"

    # Save results
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Sentiment Analysis Results - {source}\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total headlines: {len(headlines)}\n\n")

        # Calculate sentiment distribution
        sentiment_counts = {}
        for sentiment in sentiments:
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

        f.write("Sentiment Distribution:\n")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(sentiments)) * 100 if sentiments else 0
            f.write(f"  {sentiment}: {count} ({percentage:.1f}%)\n")
        f.write("\n" + "="*80 + "\n\n")

        # Write individual results
        for i, (headline, sentiment) in enumerate(zip(headlines, sentiments), 1):
            f.write(f"{i}. [{sentiment}] {headline}\n\n")

    print(f"Results saved to: {filename}")
    return filename


def main():
    """Main function to orchestrate the headline sentiment analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze sentiment of news headlines from NYT and Chicago Tribune"
    )
    parser.add_argument(
        '--source',
        choices=['nyt', 'chicago', 'both'],
        default='both',
        help='News source to fetch headlines from (default: both)'
    )
    parser.add_argument(
        '--model-path',
        default='models/svm.joblib',
        help='Path to the trained SVM model (default: models/svm.joblib)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/processed/scraped',
        help='Directory to save results (default: data/processed/scraped)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Display results without saving to file'
    )

    args = parser.parse_args()

    # Fetch headlines based on source
    all_headlines = []
    sources = []

    if args.source in ['nyt', 'both']:
        nyt_headlines = fetch_nyt_headlines()
        if nyt_headlines:
            all_headlines.extend(nyt_headlines)
            sources.extend(['NYT'] * len(nyt_headlines))

    if args.source in ['chicago', 'both']:
        chicago_headlines = fetch_chicago_tribune_headlines()
        if chicago_headlines:
            all_headlines.extend(chicago_headlines)
            sources.extend(['Chicago Tribune'] * len(chicago_headlines))

    if not all_headlines:
        print("No headlines found. Exiting.")
        sys.exit(1)

    # Generate embeddings
    embeddings = generate_embeddings(all_headlines)

    # Predict sentiments
    sentiments = predict_sentiment(embeddings, args.model_path)

    # Display summary
    print("\n" + "="*80)
    print("Sentiment Analysis Summary")
    print("="*80)

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
    print("-"*80)
    for i in range(min(5, len(all_headlines))):
        print(f"[{sentiments[i]}] {all_headlines[i][:100]}...")

    print("\nRun completed.")


if __name__ == "__main__":
    main()
