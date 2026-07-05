from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import config

DATASET_PATH = Path("data/arxiv_subset.parquet")


def main():
    df = pd.read_parquet(DATASET_PATH)

    texts = (
        df["title"].fillna("")
        + " [SEP] "
        + df["abstract"].fillna("")
    ).tolist()

    model = SentenceTransformer(config.MODEL_NAME)

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print(f"Загальна кількість оброблених текстів: {len(texts)}")
    print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
    print(f"Норма першого ембеддингу: {np.linalg.norm(embeddings[0]):.6f}")

    config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(
        config.INPUT_EMBEDDINGS,
        embeddings
    )


if __name__ == "__main__":
    main()