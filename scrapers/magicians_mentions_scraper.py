# magicians_mentions_scraper.py (robust version)
import time
import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

BASE_MAGICIANS = "https://ethereum-magicians.org/search.json?q="

def extract_repo_short(url: str) -> str:
    """Extract short repo form like 'owner/name' from GitHub URL."""
    try:
        parts = url.strip().split("github.com/")[-1].split("/")
        return "/".join(parts[:2]).lower()
    except Exception:
        return url.lower()

def fetch_with_retries(url, max_retries=3, base_sleep=5):
    """Handle 429 rate limits and retry with backoff."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:  # rate limited
                wait = base_sleep * (2 ** attempt)
                print(f"[429] Rate limited. Sleeping {wait}s before retry...")
                time.sleep(wait)
            elif r.status_code == 400:
                print(f"[400] Bad request for {url}. Skipping.")
                return None
            else:
                print(f"[{r.status_code}] Non-200 response. Skipping.")
                return None
        except Exception as e:
            print(f"[Error] {e}. Retrying...")
            time.sleep(base_sleep * (2 ** attempt))
    return None

def scrape_magicians_mentions(repo_list, sleep=1):
    results = []
    for repo in repo_list:
        query = repo.split("/")[-1]  # start with short project name
        url = f"{BASE_MAGICIANS}{query}"
        print(f"[Magicians] Searching {query}")

        data = fetch_with_retries(url)
        # fallback if query fails
        if not data:
            fallback_query = repo.replace("/", "%20")
            print(f"[Fallback] Trying full repo name: {fallback_query}")
            url = f"{BASE_MAGICIANS}{fallback_query}"
            data = fetch_with_retries(url)

        if not data:
            results.append({"repo_name": repo, "num_threads": 0, "unique_authors": 0, "last_mention": None})
            continue

        topics = data.get("topics", [])
        posts = data.get("posts", [])

        if topics:
            thread_ids = {t["id"] for t in topics}
            unique_authors = len({p["username"] for p in posts}) if posts else 0
            last_mention = max([t.get("last_posted_at", "") for t in topics if t.get("last_posted_at")])

            results.append({
                "repo_name": repo,
                "num_threads": len(thread_ids),  # capped at 50 by API
                "unique_authors": unique_authors,
                "last_mention": last_mention
            })
        else:
            results.append({"repo_name": repo, "num_threads": 0, "unique_authors": 0, "last_mention": None})

        time.sleep(sleep)  # keep a base sleep to avoid hammering server

    df = pd.DataFrame(results)

    # Add normalized columns
    if not df.empty:
        scaler = MinMaxScaler()
        for col in ["num_threads", "unique_authors"]:
            values = df[[col]].astype(float)
            df[f"{col}_norm"] = scaler.fit_transform(values)

    df.to_csv("magicians_mentions.csv", index=False)
    print(f"[Magicians] Saved {len(df)} repos → magicians_mentions.csv")
    return df

if __name__ == "__main__":
    # Load repo list from test.csv
    test_df = pd.read_csv("test.csv")
    repo_urls = test_df["repo"].dropna().tolist()
    repo_list = [extract_repo_short(url) for url in repo_urls]

    print(f"Loaded {len(repo_list)} repos from test.csv")
    scrape_magicians_mentions(repo_list)
