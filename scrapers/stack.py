import os
import re
import time
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
STACK_API_KEY = os.getenv("STACKEXCHANGE_API_KEY")

# Configuration
DATA_PATH = r"C:\Users\Path"
COMPETITION_TEST_FILE = os.path.join(DATA_PATH, "test.csv") 
ENHANCED_TEAMS_FILE = os.path.join(DATA_PATH, "DeepFunding Repos Enhanced via OpenQ - ENHANCED TEAMS.csv")
OUTPUT_FILE = os.path.join(DATA_PATH, "stackoverflow_analysis_fixed.csv")

def get_competition_repos():
    """Get only the 45 competition repositories from test.csv"""
    try:
        test_df = pd.read_csv(COMPETITION_TEST_FILE)
        
        def extract_repo_name(url):
            if pd.isna(url) or not isinstance(url, str):
                return None
            if url.startswith("https://github.com/"):
                return url[len("https://github.com/"):].lower().rstrip('/')
            return url.lower()
        
        competition_repos = set()
        test_df['repo_name'] = test_df['repo'].apply(extract_repo_name)
        competition_repos.update(test_df['repo_name'].dropna())
        
        print(f"Found {len(competition_repos)} competition repositories")
        return competition_repos
        
    except Exception as e:
        print(f"Error loading competition repos: {e}")
        return set()

def extract_search_terms(repo_name):
    """Enhanced keyword extraction for Ethereum projects"""
    if pd.isna(repo_name):
        return None
        
    repo_name = str(repo_name).lower()
    
    # Remove common prefixes/suffixes
    for suffix in ['.js', '.py', '-rs', '-go', '-rs', '-node', '-js', '-ts']:
        if repo_name.endswith(suffix):
            repo_name = repo_name[:-len(suffix)]
    
    # Handle special cases for better search
    special_cases = {
        'revm-inspectors': 'revm',
        'testcontainers-node': 'testcontainers ethereum',
        'body-parser': None,
        'mdx': None,
        'viem': 'viem ethereum',
        'hardhat': 'hardhat ethereum',
        'foundry': 'foundry ethereum',
        'ape': 'ape ethereum framework',
        'helios': 'helios ethereum',
        'alloy': 'alloy ethereum',
        'reth': 'reth ethereum',
        'lodestar': 'lodestar ethereum',
        'teku': 'teku ethereum',
        'prysm': 'prysm ethereum',
        'lighthouse': 'lighthouse ethereum',
        'nimbus-eth2': 'nimbus ethereum',
        'web3j': 'web3j ethereum',
        'web3py': 'web3.py ethereum',
        'nethereum': 'nethereum',
        'py-evm': 'py-evm ethereum',
        'ethers': 'ethers.js ethereum',
        'web3js': 'web3.js ethereum'
    }
    
    if repo_name in special_cases:
        return special_cases[repo_name]
    
    # General cleanup
    repo_name = re.sub(r'[-_]', ' ', repo_name)
    
    # Add 'ethereum' keyword for better relevance
    if repo_name and 'ethereum' not in repo_name:
        repo_name = f"{repo_name} ethereum"
    
    return repo_name.strip() if repo_name else None

def query_stackoverflow(keyword, retries=3):
    """Robust StackOverflow query with error handling"""
    if not keyword:
        return None
        
    params = {
        "order": "desc",
        "sort": "relevance",
        "tagged": "ethereum",
        "q": keyword,
        "site": "stackoverflow",
        "key": STACK_API_KEY,
        "pagesize": 50,
        "filter": "!nNPvSNdWme"
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(
                "https://api.stackexchange.com/2.3/search/advanced",
                params=params,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            if 'backoff' in data:
                time.sleep(data['backoff'] + 1)
                
            items = data.get('items', [])
            if not items:
                return {
                    'total_questions': 0,
                    'avg_score': 0,
                    'answered_ratio': 0,
                    'top_tags': ''
                }
                
            # Calculate metrics
            total_score = sum(i.get('score', 0) for i in items)
            answered_count = sum(1 for i in items if i.get('is_answered', False))
            
            # Safely extract top tags
            tag_counts = {}
            for item in items:
                for tag in item.get('tags', []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            top_tags_str = ', '.join([tag for tag, count in top_tags])
            
            return {
                'total_questions': len(items),
                'avg_score': total_score / len(items) if items else 0,
                'answered_ratio': answered_count / len(items) if items else 0,
                'top_tags': top_tags_str
            }
            
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                print(f"Failed to query '{keyword}': {str(e)}")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"Unexpected error for '{keyword}': {str(e)}")
            time.sleep(2 ** attempt)
    
    return None

def process_repository(repo_row):
    """Process a single repository with error handling"""
    try:
        repo_name = repo_row['name']
        keyword = extract_search_terms(repo_name)
        
        if not keyword:
            print(f"⚠️  Skipping {repo_name} - no valid search terms")
            return None
            
        print(f"🔍 Searching for: {keyword}")
        so_data = query_stackoverflow(keyword)
        
        if so_data is None:
            print(f"Failed to get data for {repo_name}")
            return None
            
        return {
            'owner': repo_row['owner'],
            'repo': repo_name,
            'github_url': repo_row['githubLink'],
            'search_keyword': keyword,
            **so_data
        }
        
    except Exception as e:
        print(f"Error processing {repo_row.get('name', 'unknown')}: {str(e)}")
        return None

def main():
    # Get competition repositories
    competition_repos = get_competition_repos()
    if not competition_repos:
        print(" No competition repositories found")
        return
    
    # Load enhanced teams data
    try:
        df = pd.read_csv(ENHANCED_TEAMS_FILE)
        print(f"Loaded {len(df)} repositories from enhanced teams data")
        
        # Filter to only competition repositories
        df['repo_name'] = (df['owner'] + '/' + df['name']).str.lower()
        competition_df = df[df['repo_name'].isin(competition_repos)]
        
        if competition_df.empty:
            print(" No matching repositories found in enhanced teams data")
            print("Competition repos:", competition_repos)
            return
            
        print(f"Filtered to {len(competition_df)} competition repositories")
        
        # Filter valid repositories
        repos_to_process = [
            row for _, row in competition_df.iterrows()
            if isinstance(row.get('githubLink'), str) 
            and 'github.com' in row['githubLink']
        ]
        
        print(f"Processing {len(repos_to_process)} valid competition repositories")
        
        # Process with progress bar
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:  # Reduced workers to avoid rate limits
            futures = {executor.submit(process_repository, row): row for row in repos_to_process}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing competition repos"):
                result = future.result()
                if result:
                    results.append(result)
                time.sleep(1)  # Additional rate limiting
        
        # Save results
        if results:
            result_df = pd.DataFrame(results)
            result_df.to_csv(OUTPUT_FILE, index=False)
            print(f"✅ Saved results for {len(results)} repositories to {OUTPUT_FILE}")
            
            # Show summary stats
            print("\n Summary Statistics:")
            print(f"- Total questions: {result_df['total_questions'].sum()}")
            print(f"- Average questions per repo: {result_df['total_questions'].mean():.1f}")
            print(f"- Average score: {result_df['avg_score'].mean():.1f}")
            print(f"- Average answered ratio: {result_df['answered_ratio'].mean():.1%}")
            
            # Show top 10 repos by questions
            print("\n Top 10 repositories by StackOverflow questions:")
            top_10 = result_df.nlargest(10, 'total_questions')[['repo', 'total_questions', 'avg_score']]
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                print(f"  {i:2d}. {row['repo']}: {row['total_questions']} questions (avg score: {row['avg_score']:.1f})")
                
        else:
            print(" No valid results found")
            
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()