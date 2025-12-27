import os
import logging
import inspect
from pathlib import Path
# 引入 shutil 用于清理锁文件
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_qdrant import Qdrant
from text_splitter_medical import UnifiedMedicalTextSplitter
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class KnowledgeManager:
    """知识库管理器：负责文档处理与向量存储"""

    def __init__(self, data_path: str, embedding_model, collection_name: str = "medical_db"):
        self.data_path = data_path
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.qdrant_path = "./qdrant_db"
        self.vector_store = None

        # 0. [新增] 启动前清理可能的残留锁文件 (防止上次崩溃导致的死锁)
        self._clean_stale_lock()

        self._auto_convert_pdfs()
        self._load_and_index_documents()

    def _clean_stale_lock(self):
        """清理 Qdrant 可能残留的 .lock 文件"""
        lock_file = Path(self.qdrant_path) / ".lock"
        if lock_file.exists():
            try:
                os.remove(lock_file)
                logger.warning("检测到残留的 Qdrant 锁文件，已自动清理。")
            except Exception as e:
                logger.warning(f"尝试清理锁文件失败: {e}")

    def _auto_convert_pdfs(self):
        """扫描并转换 PDF"""
        try:
            data_dir = Path(self.data_path)
            if not data_dir.exists():
                data_dir.mkdir(parents=True, exist_ok=True)
                return

            pdf_files = list(data_dir.glob("*.pdf"))
            if not pdf_files:
                return

            to_convert = []
            for pdf in pdf_files:
                txt_path = pdf.with_suffix(".txt")
                if not txt_path.exists():
                    to_convert.append(pdf)

            if not to_convert:
                return

            logger.info(f"检测到 {len(to_convert)} 个未转换的 PDF 文件，正在自动转换为 TXT...")
            try:
                from pdf_to_txt_clean import convert_pdf_to_txt
                for pdf_path in to_convert:
                    logger.info(f"→ 转换: {pdf_path.name}")
                    try:
                        convert_pdf_to_txt(str(pdf_path), output_dir=str(data_dir))
                    except Exception as e:
                        logger.error(f"转换失败 {pdf_path.name}: {e}", exc_info=True)
                logger.info("PDF 转 TXT 完成。")
            except ImportError:
                logger.warning("未找到 pdf_to_txt_clean 模块，跳过转换。")

        except Exception as e:
            logger.error(f"PDF 自动转换过程出错: {e}", exc_info=True)

    def _load_and_index_documents(self):
        """
        生产级优化：
        1. 检查本地 Qdrant 是否已有当前集合数据。
        2. 如果有，直接加载。
        3. 如果无，释放锁并执行全量构建。
        """
        logger.debug(f"Qdrant class location: {inspect.getfile(Qdrant)}")
        logger.info("加载医学文档...")

        # 1. 初始化客户端检查集合状态
        # 注意：这一步会锁定 DB
        check_client = QdrantClient(path=self.qdrant_path)
        should_reuse = False

        try:
            # 获取所有集合列表
            collections = check_client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            # 如果集合存在，且里面有数据
            if exists:
                count = check_client.count(self.collection_name).count
                if count > 0:
                    logger.info(f"发现现有向量库 '{self.collection_name}' (含 {count} 条数据)，跳过构建，直接加载。")
                    should_reuse = True
        except Exception as e:
            logger.warning(f"检查现有向量库时出错: {e}")

        # 2. 关键分支逻辑
        if should_reuse:
            # A. 复用模式：继续使用 check_client，不要关闭它
            self.vector_store = Qdrant(
                client=check_client,
                collection_name=self.collection_name,
                embeddings=self.embedding_model,
            )
            return  # 结束
        else:
            # B. 重建模式：必须先关闭 check_client 释放锁！
            logger.info("未发现有效索引，释放检查锁，准备全量构建...")
            check_client.close()
            # 这里的 close() 是关键，否则下面的 from_documents 无法获取锁

        # ================= 以下是原有构建逻辑 =================
        documents = []
        try:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)

            for file in os.listdir(self.data_path):
                full_path = os.path.join(self.data_path, file)
                if file.endswith(".txt") and os.path.isfile(full_path):
                    loader = TextLoader(full_path, encoding="utf-8")
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = file
                    documents.extend(docs)
                else:
                    logger.debug(f"跳过非 .txt 文件或目录: {file}")
        except Exception as e:
            logger.error(f"加载文档时发生错误: {e}")
            raise

        if not documents:
            logger.warning(f"未在 {self.data_path} 找到医学文档。")
            # 即便是空的，也要初始化一个空的 vector_store，否则后续调用会报错
            # 但 Qdrant.from_documents 需要至少一个文档。
            # 这里如果不做处理，直接返回可能导致后续报错。
            return

        # 切分
        chunks = []
        if documents:
            splitter = UnifiedMedicalTextSplitter(chunk_size=600, chunk_overlap=100)
            chunks = splitter.split_documents(documents)
            logger.info(f"成功加载 {len(documents)} 个文档，共生成 {len(chunks)} 个文本块")
            for i, doc in enumerate(chunks[:3]):
                preview = doc.page_content[:100].replace('\n', ' ')
                source = doc.metadata.get("section", "未知章节")
                logger.debug(f"Chunk [{i}] | 章节: {source} | 内容: {preview}...")

        # 构建向量库 (这里会创建新的 client)
        try:
            self.vector_store = Qdrant.from_documents(
                chunks,
                self.embedding_model,
                path=self.qdrant_path,
                collection_name=self.collection_name,
                force_recreate=True  # 全量构建时强制覆盖
            )
            logger.info("Qdrant 向量库构建完成。")
        except Exception as e:
            logger.error(f"Qdrant 写入失败: {e}")
            raise

    def add_document(self, text: str, source_name: str):
        """实时添加新文档到向量库"""
        logger.info(f"正在向知识库添加新文档: {source_name}")
        from langchain_core.documents import Document

        new_doc = Document(page_content=text, metadata={"source": source_name})
        splitter = UnifiedMedicalTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = splitter.split_documents([new_doc])

        if not chunks:
            logger.warning("文档内容过短或为空，未生成切片。")
            return False

        logger.info(f"生成 {len(chunks)} 个新切片，正在写入 Qdrant...")

        if self.vector_store:
            self.vector_store.add_documents(chunks)
            logger.info("增量更新成功！")
            return True
        else:
            logger.error("向量库尚未初始化，无法添加。")
            return False