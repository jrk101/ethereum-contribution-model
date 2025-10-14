import requests
import pandas as pd
from tqdm import tqdm
import os
import re
# --- CONFIGURATION ---
INPUT_CSV = "test.csv"
OUTPUT_CSV = "package_downloads.csv"

# --- REPO TO PACKAGE MAPPING ---
# This dictionary maps the GitHub repo name to its package manager name and type.
REPO_TO_PACKAGE_MAP = {
    'a16z/helios': ('helios-light-client', 'npm'),
    'alloy-rs/alloy': ('alloy', 'cargo'), # Cargo doesn't have a simple public download API, will be 0
    'apeworx/ape': ('eth-ape', 'pypi'),
    'consensys/teku': (None, None), # Java project, no central package manager
    'eth-infinitism/account-abstraction': ('@erc-4337/contracts', 'npm'),
    'dapphub/dapp': (None, None),
    'dapphub/ds-test': ('ds-test', 'npm'),
    'dapphub/seth': (None, None),
    'argotorg/fe': (None, None),
    'ethereum/go-ethereum': (None, None), # Go project
    'ethereum/py-evm': ('py-evm', 'pypi'),
    'ethereum/remix-project': ('remix-ide', 'npm'),
    'argotorg/sourcify': ('sourcify-server', 'npm'),
    'ethereum/web3.py': ('web3', 'pypi'),
    'ethers-io/ethers.js': ('ethers', 'npm'),
    'flashbots/ethers-provider-flashbots-bundle': ('ethers-provider-flashbots-bundle', 'npm'),
    'grandinetech/grandine': (None, None),
    'hyperledger-web3j/web3j': (None, None), # Java project
    'hyperledger/besu': (None, None), # Java project
    'trufflesuite/ganache': ('ganache', 'npm'),
    'trufflesuite/truffle': ('truffle', 'npm'),
    'nomicfoundation/hardhat': ('hardhat', 'npm'),
    'openzeppelin/openzeppelin-contracts': ('@openzeppelin/contracts', 'npm'),
    'safe-global/safe-smart-account': ('@safe-global/protocol-kit', 'npm'),
    'prysmaticlabs/prysm': (None, None), # Go project
    'vyperlang/vyper': ('vyper', 'pypi'),
    'chainsafe/lodestar': ('@chainsafe/lodestar', 'npm'),
    'sigp/lighthouse': (None, None), # Rust project
    'vyperlang/titanoboa': ('titanoboa', 'pypi'),
    'ethereum-attestation-service/eas-contracts': ('@ethereum-attestation-service/eas-contracts', 'npm'),
    'argotorg/hevm': (None, None),
    'ethdebug/format': (None, None),
    'argotorg/act': (None, None),
    'ethpandaops/ethereum-package': ('@ethpandaops/ethereum-package', 'npm'),
    'ethpandaops/ethereum-helm-charts': (None, None), # Helm charts
    'ethpandaops/checkpointz': ('@ethpandaops/checkpointz', 'npm'),
    'erigontech/erigon': (None, None), # Go project
    'erigontech/silkworm': (None, None), # C++ project
    'lambdaclass/lambda_ethereum_consensus': (None, None)
}

# --- HELPER FUNCTIONS ---

def get_npm_downloads(package_name):
    """Fetches the last month's download count for an NPM package."""
    try:
        url = f"https://api.npmjs.org/downloads/point/last-month/{package_name}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json().get("downloads", 0)
    except requests.exceptions.RequestException:
        pass
    return 0

def get_pypi_downloads(package_name):
    """Fetches the last month's download count for a PyPI package."""
    try:
        url = f"https://pypistats.org/api/packages/{package_name}/recent?period=last_month"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()["data"].get("last_month", 0)
    except requests.exceptions.RequestException:
        pass
    return 0
    
def extract_repo_name_from_url(url):
    """Extracts 'owner/repo' from a GitHub URL."""
    if pd.isna(url) or not isinstance(url, str):
        return None
    match = re.search(r'github\.com/([^/]+/[^/]+)', url.lower())
    if match:
        return match.group(1).replace('.git', '')
    return None

# --- MAIN SCRIPT ---

if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        print(f" Error: Input file '{INPUT_CSV}' not found.")
        exit()

    target_df = pd.read_csv(INPUT_CSV)
    target_df['repo_name'] = target_df['repo'].apply(extract_repo_name_from_url)
    target_repos = list(target_df['repo_name'].dropna())
    print(f" Loaded {len(target_repos)} repos from '{INPUT_CSV}'.")

    package_data = []

    for repo in tqdm(target_repos, desc="Fetching package downloads"):
        package_name, package_type = REPO_TO_PACKAGE_MAP.get(repo, (None, None))
        
        downloads = 0
        if package_name and package_type == 'npm':
            downloads = get_npm_downloads(package_name)
        elif package_name and package_type == 'pypi':
            downloads = get_pypi_downloads(package_name)
            
        package_data.append({
            "repo_name": repo,
            "monthly_downloads": downloads
        })
        
    results_df = pd.DataFrame(package_data)
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n Successfully saved data for {len(results_df)} repos to '{OUTPUT_CSV}'.")
    print("Top 5 projects by monthly downloads:")
    print(results_df.sort_values("monthly_downloads", ascending=False).head())