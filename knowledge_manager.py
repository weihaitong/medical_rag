import os
import logging
import inspect
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_qdrant import Qdrant
from text_splitter_medical import UnifiedMedicalTextSplitter

logger = logging.getLogger(__name__)


class KnowledgeManager:
    """知识库管理器：负责文档处理与向量存储"""

    def __init__(self, data_path: str, embedding_model, collection_name: str = "medical_db"):
        self.data_path = data_path
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.qdrant_path = "./qdrant_db"
        self.vector_store = None

        self._auto_convert_pdfs()
        self._load_and_index_documents()

    def _auto_convert_pdfs(self):
        """扫描并转换 PDF"""
        try:
            from pathlib import Path
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
            # 动态引用，避免循环依赖或强制依赖
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
        """加载 TXT，切分，写入 Qdrant"""
        logger.debug(f"Qdrant class location: {inspect.getfile(Qdrant)}")
        logger.info("加载医学文档...")

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

        # 构建向量库
        try:
            self.vector_store = Qdrant.from_documents(
                chunks if chunks else [],  # 避免空列表报错
                self.embedding_model,
                path=self.qdrant_path,
                collection_name=self.collection_name,
                force_recreate=False
            )
            logger.info("Qdrant 向量库构建完成。")
        except Exception as e:
            logger.error(f"Qdrant 写入失败: {e}")
            raise

    # [新增功能] 动态添加单篇文档到向量库
    def add_document(self, text: str, source_name: str):
        """
        实时添加新文档到向量库，无需重启
        """
        logger.info(f"正在向知识库添加新文档: {source_name}")

        # 1. 封装为 Document 对象
        new_doc = Document(page_content=text, metadata={"source": source_name})

        # 2. 切分 (保持和初始化时一致的切分逻辑)
        splitter = UnifiedMedicalTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = splitter.split_documents([new_doc])

        if not chunks:
            logger.warning("文档内容过短或为空，未生成切片。")
            return False

        logger.info(f"生成 {len(chunks)} 个新切片，正在写入 Qdrant...")

        # 3. 增量写入向量库
        if self.vector_store:
            self.vector_store.add_documents(chunks)
            logger.info("增量更新成功！")
            return True
        else:
            logger.error("向量库尚未初始化，无法添加。")
            return False