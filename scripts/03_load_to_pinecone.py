import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone

import config
from helpers.pinecone_helper import ensure_index

load_dotenv()

BATCH_SIZE = 200

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = ensure_index(pc, config.INDEX_NAME)

df = pd.read_parquet(config.INPUT_PARQUET)
embeddings = np.load(config.INPUT_EMBEDDINGS)

for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Uploading to Pinecone"):
    end = min(start + BATCH_SIZE, len(df))
    batch = []

    for i in range(start, end):
        row = df.iloc[i]

        vector = {
            "id": f"paper_{i}",
            "values": embeddings[i].tolist(),
            "metadata": {
                "arxiv_id": str(row["id"]),
                "title": str(row["title"]),
                "abstract": str(row["abstract"])[:500],
                "authors": str(row["authors"])[:200],
                "year": int(row["year"]),
                "category": str(row["category"]),
            },
        }

        batch.append(vector)

    index.upsert(vectors=batch)

stats = index.describe_index_stats()

print("Завантаження завершено")
print(f"Загальна кількість векторів в індексі: {stats['total_vector_count']}")
