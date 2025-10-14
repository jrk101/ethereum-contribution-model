import re
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Set paths
INPUT_CSV = r"C:\Users\Path\test.csv"
OUTPUT_CSV = r"C:\Users\Path\repo_to_contract.csv"

# Regex to find Ethereum contract addresses
ETH_ADDRESS_REGEX = r'0x[a-fA-F0-9]{40}'

def extract_addresses_from_text(text):
    """Extract all Ethereum addresses from a text blob"""
    return re.findall(ETH_ADDRESS_REGEX, text)

def fetch_github_file(repo_url, filepath):
    try:
        parts = repo_url.rstrip('/').split('/')
        owner, repo = parts[-2], parts[-1]
        branches = ['main', 'master']
        for branch in branches:
            raw_url = f'https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filepath}'
            res = requests.get(raw_url, timeout=10)
            if res.status_code == 200:
                return res.text
    except:
        return ""
    return ""

def process_repo(row):
    repo_url = row['repo']
    result = {'repo': repo_url, 'contract_address': ''}

    # Try README
    readme = fetch_github_file(repo_url, 'README.md')
    addresses = extract_addresses_from_text(readme)

    # If not found in README, check contracts directory
    if not addresses:
        for filename in ['contracts/Contract.sol', 'contracts/index.sol', 'contracts/main.sol']:
            content = fetch_github_file(repo_url, filename)
            addresses += extract_addresses_from_text(content)
            if addresses:
                break

    result['contract_address'] = addresses[0] if addresses else ''
    return result

# Load CSV
df = pd.read_csv(INPUT_CSV)

if 'repo' not in df.columns:
    raise ValueError("CSV must contain a 'repo' column with GitHub URLs")

# Optional: for testing, use smaller subset
# df = df.head(100)

print(" Extracting contract addresses from GitHub repos (multi-threaded)...")

# Use multithreading for faster processing
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(tqdm(executor.map(process_repo, df.to_dict(orient='records')), total=len(df)))

# Save to CSV
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"Saved contract address mapping to: {OUTPUT_CSV}")
