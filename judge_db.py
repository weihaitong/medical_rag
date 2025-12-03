import os
from qdrant_client import QdrantClient, models


def init_qdrant(
    collection_name="medical_db",
    vector_size=1024,
    distance=models.Distance.COSINE,
):
    """
    初始化本地持久化 Qdrant（Embedded）。
    - 判断 collection 是否存在，若存在则加载
    - 若不存在则创建
    """

    base_path = f"qdrant_db/collection/{collection_name}"
    os.makedirs(base_path, exist_ok=True)

    client = QdrantClient(path=base_path)

    # ---- 正确判断 collection 是否存在 ----
    def collection_exists():
        try:
            client.get_collection(collection_name)
            return True
        except Exception:
            return False

    # ---- 根据存在情况决定创建或加载 ----
    if collection_exists():
        print(f"→ collection 已存在，加载本地数据库: {base_path}")
    else:
        print(f"→ collection 不存在，创建新的 collection: {collection_name}")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=distance,
            ),
        )

    return client


# ---- 独立运行测试 ----
if __name__ == "__main__":
    client = init_qdrant("medical_db")

    # 打印所有 collections（注意：空 collection 可能不显示）
    try:
        col = client.get_collection("medical_db")
        print("已成功加载 collection:", col)
    except:
        print("无法加载 collection（意外情况）")
