import requests
import pandas as pd
from tqdm import tqdm
import time
import os
import json
import re

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
INPUT_CSV_FILES = ["train.csv", "test.csv"]
OUTPUT_CSV = "enhanced_github_data.csv"
TEMP_FILE = "enhanced_temp.json"
SKIPPED_FILE = "skipped_repos.csv"

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# === HELPERS ===
def fetch_with_retry(url, retries=2, backoff=10):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 403:
                print(" Rate limit hit. Sleeping 60s...")
                time.sleep(60)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}. Retrying ({attempt + 1}/{retries}) in {backoff}s...")
            time.sleep(backoff)
    print(f" Max retries exceeded for: {url}")
    return None


def estimate_commit_count(repo_full_name):
    commits_url = f"https://api.github.com/repos/{repo_full_name}/commits?per_page=1"
    r = fetch_with_retry(commits_url)
    if not r:
        return 0
    link = r.headers.get("Link", "")
    if 'rel="last"' in link:
        match = re.search(r"&page=(\d+)>; rel=\"last\"", link)
        if match:
            return int(match.group(1))
    return 1


def count_pull_requests(repo_full_name, state):
    """Count pull requests by state: open, closed"""
    url = f"https://api.github.com/search/issues?q=repo:{repo_full_name}+is:pr+state:{state}"
    r = fetch_with_retry(url)
    if r and "total_count" in r.json():
        return r.json()["total_count"]
    return 0


def fetch_releases(repo_full_name):
    url = f"https://api.github.com/repos/{repo_full_name}/releases"
    r = fetch_with_retry(url)
    if not r:
        return 0, ""
    releases = r.json()
    if isinstance(releases, list) and len(releases) > 0:
        latest_date = releases[0].get("published_at", "")
        return len(releases), latest_date
    return 0, ""


def fetch_repo_data(repo_full_name):
    base_url = f"https://api.github.com/repos/{repo_full_name}"
    r = fetch_with_retry(base_url)
    if r is None:
        return None
    repo_data = r.json()

    # Contributors
    contrib_url = f"https://api.github.com/repos/{repo_full_name}/contributors?per_page=1&anon=true"
    r2 = fetch_with_retry(contrib_url)
    contributors = 0
    if r2 and isinstance(r2.json(), list):
        link = r2.headers.get("Link", "")
        if 'rel="last"' in link:
            match = re.search(r"&page=(\d+)>; rel=\"last\"", link)
            if match:
                contributors = int(match.group(1))
        else:
            contributors = len(r2.json())

    # Commits
    commit_count = estimate_commit_count(repo_full_name)

    # Pull Request Activity (merged omitted to avoid errors)
    open_prs = count_pull_requests(repo_full_name, "open")
    closed_prs = count_pull_requests(repo_full_name, "closed")

    # Release Info
    releases_count, latest_release_date = fetch_releases(repo_full_name)

    # License
    license_info = repo_data.get("license", {}) or {}
    license_id = license_info.get("spdx_id") or license_info.get("name", "")

    return {
        "repo": repo_full_name,
        "stars": repo_data.get("stargazers_count", 0),
        "forks": repo_data.get("forks_count", 0),
        "open_issues": repo_data.get("open_issues_count", 0),
        "watchers": repo_data.get("subscribers_count", 0),
        "last_pushed": repo_data.get("pushed_at", ""),
        "created_at": repo_data.get("created_at", ""),
        "contributors": contributors,
        "commitCount": commit_count,
        "language": repo_data.get("language", ""),
        "size": repo_data.get("size", 0),
        "archived": repo_data.get("archived", False),
        "fork": repo_data.get("fork", False),
        "open_pull_requests": open_prs,
        "closed_pull_requests": closed_prs,
        "releases_count": releases_count,
        "latest_release": latest_release_date,
        "license": license_id
    }


# === MAIN ===
if __name__ == "__main__":
    # Step 1: Load repos
    all_repos = set()
    for file in INPUT_CSV_FILES:
        if not os.path.exists(file):
            continue
        df = pd.read_csv(file)
        for col in ["repo", "repo_a", "repo_b"]:
            if col in df.columns:
                extracted = df[col].dropna().str.extract(r"github\.com/([^ )]+)")[0].dropna()
                all_repos.update(extracted)

    print(f"Total unique repos found: {len(all_repos)}")

    # Step 2: Load temp data
    repo_data = []
    processed = set()
    if os.path.exists(TEMP_FILE):
        with open(TEMP_FILE, "r") as f:
            repo_data = json.load(f)
            processed = set(r["repo"] for r in repo_data)

    skipped_repos = []

    # Step 3: Fetch each repo
    for repo in tqdm(sorted(all_repos)):
        if repo in processed:
            continue
        try:
            data = fetch_repo_data(repo)
            if data:
                repo_data.append(data)
                with open(TEMP_FILE, "w") as f:
                    json.dump(repo_data, f)
            else:
                skipped_repos.append(repo)
        except Exception as e:
            print(f" Error fetching {repo}: {e}")
            skipped_repos.append(repo)
        time.sleep(0.7)  # Rate-limit friendly

    # Step 4: Save outputs
    df = pd.DataFrame(repo_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f" Data saved to {OUTPUT_CSV}")

    if skipped_repos:
        pd.DataFrame(skipped_repos, columns=["repo"]).to_csv(SKIPPED_FILE, index=False)
        print(f" Skipped {len(skipped_repos)} repos. Logged to {SKIPPED_FILE}")
