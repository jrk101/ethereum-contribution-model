import os
import time
import base64
import requests
import pandas as pd
from tqdm import tqdm
from langdetect import detect, LangDetectException

# ------------------- CONFIG -------------------
GITHUB_TOKEN = ""
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}
OUT_RAW = "raw_readmes.csv"
OUT_CLEAN = "clean_readmes.csv"
OUT_FEATURES = "readme_features.csv"
LOG_FILE = "readme_errors.log"
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 1  # in seconds
# ----------------------------------------------

def clean_repo_url(url):
    url = str(url).strip().replace(".git", "")
    if url.startswith("https://github.com/"):
        url = url[len("https://github.com/"):]
    return url.lower()

def fetch_readme(repo):
    urls = [
        f"https://api.github.com/repos/{repo}/readme",
        f"https://api.github.com/repos/{repo}/contents/README.md"
    ]
    for url in urls:
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                content = response.json().get("content", "")
                if response.json().get("encoding") == "base64":
                    decoded = base64.b64decode(content).decode(errors="ignore").strip()
                    return decoded
            elif response.status_code == 404:
                return None
            elif response.status_code == 403 and "rate limit" in response.text.lower():
                print("⏳ Rate limit hit. Sleeping for 60 seconds...")
                time.sleep(60)
        except Exception as e:
            print(f"⚠️ Error fetching {repo}: {e}")
            time.sleep(3)
    return None

def is_valid_readme(text):
    if not text:
        return False
    if "<html" in text.lower():
        return False
    if len(text) < 50 or len(text) > 50000:
        return False
    try:
        lang = detect(text)
        if lang != "en":
            return False
    except LangDetectException:
        return False
    return True

def extract_features(repo, text):
    return {
        "repo": repo,
        "readme_length": len(text),
        "readme_lines": text.count("\n"),
        "has_code_blocks": int("```" in text),
        "has_sections": int("# " in text or "## " in text),
        "starts_with_badge": int(text.strip().lower().startswith("![badge") or "[![badge" in text[:200]),
    }

def load_unique_repos():
    repos = set()
    for file in ["train.csv", "test.csv"]:
        if os.path.exists(file):
            df = pd.read_csv(file)
            for col in ["repo", "repo_a", "repo_b"]:
                if col in df.columns:
                    repos.update(df[col].dropna().map(clean_repo_url))
    return sorted(repos)

def main():
    repos = load_unique_repos()
    print(f"📦 Total unique repos to fetch README from: {len(repos)}")

    raw_data = []
    clean_data = []
    feature_data = []
    failed_repos = []

    for repo in tqdm(repos):
        for attempt in range(MAX_RETRIES):
            readme = fetch_readme(repo)
            if readme is not None:
                raw_data.append({"repo": repo, "readme": readme})
                if is_valid_readme(readme):
                    clean_data.append({"repo": repo, "readme": readme})
                    feature_data.append(extract_features(repo, readme))
                break
            else:
                time.sleep(SLEEP_BETWEEN_CALLS)
        else:
            failed_repos.append(repo)

    # Save results
    pd.DataFrame(raw_data).to_csv(OUT_RAW, index=False)
    pd.DataFrame(clean_data).to_csv(OUT_CLEAN, index=False)
    pd.DataFrame(feature_data).to_csv(OUT_FEATURES, index=False)

    if failed_repos:
        with open(LOG_FILE, "w") as f:
            for repo in failed_repos:
                f.write(repo + "\n")
        print(f"⚠️ Failed to fetch {len(failed_repos)} READMEs. Logged to {LOG_FILE}")
    else:
        print("✅ All READMEs fetched successfully.")

if __name__ == "__main__":
    main()
