import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# 请确保您的 embedding_size 与您的模型匹配 (例如 BGE-small 是 384)
EMBEDDING_SIZE = 384 

# 1. 初始化客户端到本地路径
client = QdrantClient(path="./qdrant_db")
COLLECTION_NAME = "test_collection"

# 2. 创建一个 Collection
print(f"创建 Collection: {COLLECTION_NAME}")
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
)

# 3. 写入一些点
print("写入测试点...")
client.upsert(
    collection_name=COLLECTION_NAME,
    wait=True,
    points=[
        PointStruct(id=1, vector=[0.1]*EMBEDDING_SIZE, payload={"text": "Test 1"}),
        PointStruct(id=2, vector=[0.2]*EMBEDDING_SIZE, payload={"text": "Test 2"}),
    ],
)
print("纯 Qdrant 客户端写入完成。")
# 检查新结构
print("--- 检查新结构 ---")
os.system('ls -ld ./qdrant_db/collections/test_collection')
os.system('ls -ld ./qdrant_db/collections/test_collection/*')
