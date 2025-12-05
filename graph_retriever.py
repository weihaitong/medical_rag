# graph_retriever.py
import logging
import os
from typing import List, Dict
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("未安装 neo4j 包，图数据库功能将被禁用。请运行: pip install neo4j")


class GraphRetriever:
    def __init__(self):
        self.driver = None
        if not NEO4J_AVAILABLE:
            logger.info("Neo4j 依赖缺失，图检索器未启用。")
            return

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j 图数据库连接成功。")
        except Exception as e:
            logger.warning(f"Neo4j 连接失败，图检索器将返回空结果: {e}")
            if self.driver:
                self.driver.close()
            self.driver = None

    def retrieve(self, query: str) -> List[Document]:
        """
        从 Neo4j 检索与 query 相关的医学关系，返回 LangChain Document 列表。
        每个 Document 的 metadata 包含 source="neo4j"。
        """
        if not self.driver:
            return []

        # 简单关键词提取（可后期替换为 NER）
        keywords = [w.strip() for w in query.split() if len(w.strip()) > 1]
        if not keywords:
            return []

        cypher = """
        MATCH (n)-[r]->(m)
        WHERE ANY(kw IN $keywords 
                  WHERE toLower(n.name) CONTAINS toLower(kw) 
                  OR toLower(m.name) CONTAINS toLower(kw))
        RETURN DISTINCT 
            n.name AS source,
            type(r) AS relation,
            m.name AS target,
            coalesce(r.description, '') AS desc
        LIMIT 5
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher, keywords=keywords)
                docs = []
                for record in result:
                    src = record["source"]
                    rel = record["relation"]
                    tgt = record["target"]
                    desc = record["desc"]
                    if desc.strip():
                        text = f"{src} {rel} {tgt}：{desc}"
                    else:
                        text = f"{src} {rel} {tgt}。"
                    docs.append(Document(page_content=text, metadata={"source": "neo4j"}))
                return docs
        except Exception as e:
            logger.error(f"图数据库查询出错: {e}")
            return []

    def __del__(self):
        if self.driver:
            self.driver.close()