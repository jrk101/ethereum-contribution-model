import pandas as pd
from sentence_transformers import SentenceTransformer

# Load cleaned readmes
readmes_df = pd.read_csv("clean_readmes.csv")
readmes_df = readmes_df.dropna(subset=["repo", "readme"])
readmes_df = readmes_df[readmes_df["readme"].str.strip().astype(bool)]

# Initialize embedding model (free, local)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed readmes in batches
batch_size = 64
embeddings = []
for i in range(0, len(readmes_df), batch_size):
    batch = readmes_df["readme"].iloc[i:i+batch_size].tolist()
    emb = model.encode(batch, show_progress_bar=True)
    embeddings.extend(emb)

# Convert embeddings to DataFrame with proper column names
embeddings_df = pd.DataFrame(embeddings)
embeddings_df.columns = [f"readme_emb_{i}" for i in range(embeddings_df.shape[1])]
embeddings_df["repo"] = readmes_df["repo"].values

# Move repo column to first position
columns = ["repo"] + [col for col in embeddings_df.columns if col != "repo"]
embeddings_df = embeddings_df[columns]

# Save to CSV
embeddings_df.to_csv("readme_embeddings.csv", index=False)
print(f"Embeddings saved to readme_embeddings.csv with {embeddings_df.shape[1]-1} dimensions")