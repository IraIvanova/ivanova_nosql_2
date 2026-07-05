from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

INPUT_PARQUET = DATA_DIR / "arxiv_subset.parquet"
INPUT_EMBEDDINGS = EMBEDDINGS_DIR / "embeddings.npy"

# Model
MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

# Pinecone
INDEX_NAME = "arxiv-papers"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# Chunking
INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"
