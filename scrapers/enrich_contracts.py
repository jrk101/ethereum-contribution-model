import os
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

def get_tx_count_and_first_tx(address):
    url = "https://api.etherscan.io/api"
    params = {
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "sort": "asc",
        "apikey": ETHERSCAN_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data["status"] == "1":
            txs = data["result"]
            return len(txs), txs[0]['timeStamp']
    except Exception as e:
        print(f" Error fetching txlist for {address}: {e}")
    return 0, None

def get_eth_balance(address):
    url = "https://api.etherscan.io/api"
    params = {
        "module": "account",
        "action": "balance",
        "address": address,
        "tag": "latest",
        "apikey": ETHERSCAN_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data["status"] == "1":
            return int(data["result"]) / 1e18  # Convert Wei to ETH
    except Exception as e:
        print(f" Error fetching balance for {address}: {e}")
    return 0.0

def get_contract_code_length(address):
    url = "https://api.etherscan.io/api"
    params = {
        "module": "contract",
        "action": "getsourcecode",
        "address": address,
        "apikey": ETHERSCAN_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data["status"] == "1":
            code = data["result"][0].get("SourceCode", "")
            return len(code)
    except Exception as e:
        print(f" Error fetching source code for {address}: {e}")
    return 0

def enrich_contracts(input_csv="repo_to_contract.csv", output_csv="contract_enriched.csv"):
    df = pd.read_csv(input_csv)
    df["has_contract"] = df["contract_address"].apply(lambda x: int(pd.notna(x) and isinstance(x, str) and len(x.strip()) > 0))

    # Initialize new columns
    df["tx_count"] = 0
    df["first_tx_time"] = None
    df["eth_balance"] = 0.0
    df["contract_code_length"] = 0

    print(" Fetching on-chain data from Etherscan...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        addr = row["contract_address"]
        if row["has_contract"]:
            tx_count, first_tx = get_tx_count_and_first_tx(addr)
            eth_balance = get_eth_balance(addr)
            code_len = get_contract_code_length(addr)

            df.at[i, "tx_count"] = tx_count
            df.at[i, "first_tx_time"] = first_tx
            df.at[i, "eth_balance"] = eth_balance
            df.at[i, "contract_code_length"] = code_len

    df.to_csv(output_csv, index=False)
    print(f"Saved enriched data to {output_csv}")

if __name__ == "__main__":
    enrich_contracts()
