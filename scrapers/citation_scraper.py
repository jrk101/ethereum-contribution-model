import os
import time
import csv
import requests
import feedparser
import pandas as pd
from tqdm import tqdm

# Input/output paths
RAW_READMES_PATH = "C:/Users/Path/raw_readmes.csv"
OUTPUT_PATH = "C:/Users/Path/arxiv_scores.csv"

# Check if input file exists
if not os.path.exists(RAW_READMES_PATH):
    print(f"Error: Input file not found at {RAW_READMES_PATH}")
    print("Please run the README fetcher script first!")
    exit()

# Load repos
try:
    df = pd.read_csv(RAW_READMES_PATH)
    repos = df['repo'].tolist()
    print(f" Loaded {len(repos)} repositories from {RAW_READMES_PATH}")
except Exception as e:
    print(f" Error loading CSV: {e}")
    exit()

# Clean to avoid duplicates
processed = set()
if os.path.exists(OUTPUT_PATH):
    try:
        existing_df = pd.read_csv(OUTPUT_PATH)
        processed = set(existing_df['repo'].tolist())
        print(f" Found {len(processed)} already processed repositories")
    except:
        print("  Could not read existing output file, starting fresh")

# Initialize output file
if not os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['repo', 'arxiv_hits'])
    print(" Created new output file")

# Function to query arXiv
def query_arxiv(repo_name):
    query = repo_name.split("/")[-1]  # Get just the project name
    print(f"   🔍 Searching arXiv for: {query}")
    
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": 50,
    }
    url = base_url + "&".join([f"{k}={v}" for k, v in params.items()])

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        hits = len(feed.entries)
        print(f"    Found {hits} papers for {repo_name}")
        return hits
    except Exception as e:
        print(f"    Error fetching {repo_name}: {e}")
        return -1

print(f"Starting arXiv search for {len(repos)} repositories...")
print("=" * 50)

# Fetch and save
new_entries = 0
with tqdm(total=len(repos), desc="Processing repositories") as pbar:
    for repo in repos:
        if repo in processed:
            pbar.update(1)
            pbar.set_postfix({"status": "skipped"})
            continue
            
        hits = query_arxiv(repo)
        
        with open(OUTPUT_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([repo, hits])
        
        new_entries += 1
        pbar.update(1)
        pbar.set_postfix({"status": "processed", "new": new_entries})
        time.sleep(3)  # Be polite to the server

print("=" * 50)
print(f" Complete! Processed {new_entries} new repositories")
print(f"Results saved to: {OUTPUT_PATH}")

# Show preview of results
if os.path.exists(OUTPUT_PATH) and new_entries > 0:
    print("\n Preview of results:")
    results_df = pd.read_csv(OUTPUT_PATH)
    print(results_df.tail(min(10, len(results_df))))