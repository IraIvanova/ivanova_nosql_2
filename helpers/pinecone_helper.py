import time

from pinecone import Pinecone, ServerlessSpec


def ensure_index(
    pc: Pinecone,
    index_name: str,
    dimension: int = 768,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
):
    existing_indexes = [index["name"] for index in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud=cloud,
                region=region,
            ),
        )

        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    return pc.Index(index_name)