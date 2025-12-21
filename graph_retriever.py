# graph_retriever.py
import logging
import os
from typing import List, Set, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# === Neo4j 支持 ===
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("未安装 neo4j 包，图数据库功能将被禁用。请运行: pip install neo4j")

# === 医学 NER 支持 ===
try:
    from medical_ner import NER
    NER_AVAILABLE = True
except ImportError as e:
    NER_AVAILABLE = False
    logger.warning(f"医学 NER 模块不可用（{e}），图检索将跳过实体识别。")


class GraphRetriever:
    def __init__(self):
        self.driver = None
        self.ner_model: Optional[NER] = None

        # 初始化 Neo4j
        if not NEO4J_AVAILABLE:
            logger.info("Neo4j 依赖缺失，图检索器未启用。")
            return

        uri = os.getenv("NEO4J_URI", "bolt://192.168.43.225:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "Neo4j9527")

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

        # 初始化 NER 模型（仅当 Neo4j 可用时加载）
        if self.driver and NER_AVAILABLE:
            try:
                self.ner_model = NER("lixin12345/chinese-medical-ner")
            except Exception as e:
                logger.error(f"医学 NER 模型初始化失败: {e}")
                self.ner_model = None

    def _extract_medical_entities(self, query: str) -> List[str]:
        """
        提取 Drug 和 Disease 实体，并拼接为完整词。
        """
        if not self.ner_model or not query:
            return []

        try:
            raw_entities = self.ner_model.ner(query)
            keywords: Set[str] = set()

            for ent in raw_entities:
                ent_type = ent.get("type", "")
                tokens = ent.get("tokens", [])
                if not tokens:
                    continue

                entity_text = "".join(tokens).strip()
                # 只保留核心医学实体
                if ent_type in {"Drug", "DiseaseNameOrComprehensiveCertificate"} and len(entity_text) >= 2:
                    keywords.add(entity_text)

            result = list(keywords)
            logger.debug(f"NER 提取医学实体: {result}")
            return result

        except Exception as e:
            logger.error(f"医学实体提取失败: {e}")
            return []

    def retrieve(self, query: str) -> List[Document]:
        """
        从 Neo4j 检索与 query 相关的医学关系。
        """
        if not self.driver:
            return []

        keywords = self._extract_medical_entities(query)
        if not keywords:
            logger.debug("未提取到有效医学实体，跳过图数据库查询。")
            return []

        cypher = """
        MATCH (n)-[r]->(m)
        WHERE n.name IN $keywords OR m.name IN $keywords
        RETURN DISTINCT
            n.name AS source,
            type(r) AS relation,
            m.name AS target,
            coalesce(r.description, '') AS desc
        LIMIT 15
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
                    text = f"{src} {rel} {tgt}：{desc}" if desc.strip() else f"{src} {rel} {tgt}。"
                    docs.append(Document(page_content=text, metadata={"source": "neo4j"}))
                logger.debug(f"图数据库返回 {len(docs)} 条结果。")
                return docs
        except Exception as e:
            logger.error(f"图数据库查询出错: {e}")
            return []

    def __del__(self):
        if self.driver:
            self.driver.close()