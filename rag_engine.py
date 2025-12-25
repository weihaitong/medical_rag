# rag_engine.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import logging
from typing import List, Optional
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.documents import Document
from langchain_qdrant import Qdrant
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import torch
import time
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import ChatPromptValue
from ollama import OllamaClient
import json
import re
import inspect
from text_splitter_medical import UnifiedMedicalTextSplitter
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Phi3ForCausalLM,
    pipeline
)
# 配置日志器（模块级）
logger = logging.getLogger(__name__)
logger.info(f"HF_ENDPOINT set to: {os.environ.get('HF_ENDPOINT')}")
logger.setLevel(logging.DEBUG)
# 禁用部分 noisy 日志（可选）
# logging.getLogger("langchain").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

#导入图库
try:
    from graph_retriever import GraphRetriever
    GRAPH_RETRIEVER_AVAILABLE = True
except ImportError:
    GRAPH_RETRIEVER_AVAILABLE = False
    logger.warning("未找到 graph_retriever.py，图检索功能将被禁用。")

def get_torch_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"   # Apple Silicon
    else:
        return "cpu"

def get_dtype():
    if torch.cuda.is_available():
        return torch.float16
    elif torch.backends.mps.is_available():
        return torch.float16   # MPS 支持 float16
    else:
        return torch.float32   # CPU 回退 float32

class MedicalRAG:
    def __init__(self, data_path: str = "data/", collection_name: str = "medical_db"):
        self.data_path = data_path
        self.qdrant_path = "./qdrant_db"
        self._auto_convert_pdfs()
        self.collection_name = collection_name
        self._init_models()
        #
        self._init_ollama()
        self._init_rewrite_llm()#必须在主模型后，如不设置重写模型可以使用主模型
        # ===== 初始化图检索器 =====
        self.graph_retriever = GraphRetriever() if GRAPH_RETRIEVER_AVAILABLE else None

        try:
            self._load_and_index_documents()
        except Exception as e:
            logger.error(f"向量库构建失败: {e}", exc_info=True)
            raise
        self._build_rag_chain()

    def _auto_convert_pdfs(self):
        """
        扫描 data_path 目录，将所有 .pdf 文件转换为 .txt（如果尚未转换）。
        转换后的文件保存在同一目录，文件名保持一致（仅扩展名变化）。
        """
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

            from pdf_to_txt_clean import convert_pdf_to_txt

            for pdf_path in to_convert:
                logger.info(f"→ 转换: {pdf_path.name}")
                try:
                    convert_pdf_to_txt(str(pdf_path), output_dir=str(data_dir))
                except Exception as e:
                    logger.error(f"转换失败 {pdf_path.name}: {e}", exc_info=True)

            logger.info("PDF 转 TXT 完成。")

        except Exception as e:
            logger.error(f"PDF 自动转换过程出错: {e}", exc_info=True)

    def _init_ollama(self):
        logger.info("初始化 OllamaClient...")
        self.ollama = OllamaClient(
            model="qwen2:7b-instruct-q5_K_M",
            base_url="http://localhost:11434",
            timeout=120,
        )

    def _init_models(self):
        logger.info("加载嵌入模型 (BAAI/bge-m3)...")
        try:
            model_path = "models/bge-m3"
            if os.path.exists(model_path):
                logger.info(f"→ 使用本地模型: {model_path}")
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=model_path,
                    model_kwargs={"device": get_torch_device()},
                    encode_kwargs={"normalize_embeddings": True}
                )
            else:
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-m3",
                    model_kwargs={"device": get_torch_device()},
                    encode_kwargs={"normalize_embeddings": True}
                )
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}", exc_info=True)
            logger.warning("→ 使用轻量级备用模型: sentence-transformers/all-MiniLM-L6-v2")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": get_torch_device()},
            )

        logger.info("加载重排器 (BAAI/bge-reranker-v2-m3)...")
        try:
            reranker_path = "models/bge-reranker-v2-m3"
            if os.path.exists(reranker_path):
                logger.info(f"→ 使用本地重排器模型: {reranker_path}")
                self.reranker = HuggingFaceCrossEncoder(model_name=reranker_path)
            else:
                self.reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        except Exception as e:
            logger.error(f"重排器加载失败: {e}", exc_info=True)
            logger.warning("→ 使用备用重排器: cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.reranker = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

        logger.info("加载本地 LLM (Qwen/Qwen2.5-0.5B-Instruct)...")
        try:
            llm_path = "models/qwen2.5-0.5b"
            if os.path.exists(llm_path):
                logger.info(f"→ 使用本地LLM: {llm_path}")
                model = AutoModelForCausalLM.from_pretrained(
                    llm_path,
                    torch_dtype=get_dtype(),
                    device_map="auto",
                    trust_remote_code=True
                )
                tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
            else:
                model_name = "Qwen/Qwen2.5-0.5B-Instruct"
                logger.info(f"→ 从Hugging Face下载LLM: {model_name} (这可能需要几分钟)")
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=get_dtype(),
                    device_map="auto",
                    trust_remote_code=True
                )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
                repetition_penalty=1.2
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            logger.error(f"LLM加载失败: {e}", exc_info=True)
            logger.warning("→ 使用超轻量级模型: microsoft/Phi-3-mini-4k-instruct (需要在线下载)")
            try:
                model = Phi3ForCausalLM.from_pretrained(
                    "microsoft/Phi-3-mini-4k-instruct",
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True,
                )
                tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=256,
                    temperature=0.3
                )
                self.llm = HuggingFacePipeline(pipeline=pipe)
            except Exception as e2:
                logger.error(f"备用LLM加载失败: {e2}", exc_info=True)
                logger.critical("无法加载任何语言模型。请检查网络连接或创建 models/ 目录并放置本地模型。")
                raise

    def _init_rewrite_llm(self):
        """
        Query Rewrite LLM：
        - 优先使用 FastAPI OpenAI-compatible Server
        - 失败 / 超时 / 异常时自动降级为本地 HuggingFacePipeline
        """

        # ========= 1. 本地 fallback（必须成功） =========
        def load_local_rewrite_llm():
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=get_dtype(),
                device_map="auto",
                trust_remote_code=True
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=80,
                #temperature=0.0,
                do_sample=False
            )
            return HuggingFacePipeline(pipeline=pipe)

        try:
            local_llm = load_local_rewrite_llm()
            logger.info("Rewrite LLM 本地 fallback 加载成功")
        except Exception:
            logger.exception("Rewrite 本地模型加载失败，直接复用主 LLM")
            self.rewrite_llm = self.llm
            return

        # ========= 2. FastAPI 远程 Rewrite LLM =========
        try:
            from fastapi_rewrite_llm import FastAPIRewriteLLM

            remote_llm = FastAPIRewriteLLM(
                endpoint="http://127.0.0.1:8000/v1/chat/completions",
                model="Qwen/Qwen2.5-0.5B-Instruct",
                timeout=2.0
            )

            # ========= 3. 远程优先 + 本地降级 =========
            class RewriteWithFallback(Runnable):
                def invoke(self, input, config=None, **kwargs) -> str:
                    """
                    Rewrite 查询：
                    1. 优先调用远端 FastAPI LLM
                    2. 严格解析 OpenAI ChatCompletion 结构
                    3. 失败时自动降级本地模型
                    """
                    logger.debug(
                        "RewriteWithFallback input type=%s, input=%s",
                        type(input),
                        input
                    )
                    # ===== 1. 保留 ChatPromptValue 的完整结构 =====
                    if isinstance(input, ChatPromptValue):
                        prompt_input = input
                        logger.debug(
                            "Rewrite received ChatPromptValue, messages=%d",
                            len(input.to_messages())
                        )
                    else:
                        prompt_input = ChatPromptValue.from_messages([
                            ("human", str(input))
                        ])
                        logger.debug(
                            "Rewrite extracted prompt_input from str ChatPromptValue: %s",
                            prompt_input
                        )

                    # ===== 2. 远端优先 =====
                    try:
                        start = time.time()
                        content = remote_llm.invoke(prompt_input)
                        latency = time.time() - start
                        logger.info(
                            "Rewrite remote success, latency=%.2fs, resp_type=%s, output=%s",
                            latency,
                            type(content),
                            content
                        )
                        if isinstance(content, BaseMessage):
                            text = content.content
                        else:
                            text = str(content)

                        return text.strip()

                    except Exception as e:
                        if isinstance(prompt_input, ChatPromptValue):
                            local_prompt = prompt_input.to_string()
                        else:
                            local_prompt = str(prompt_input)

                        logger.warning("Rewrite fallback to local, reason=%s, query=%s", e, local_prompt)
                        local_out = local_llm.invoke(local_prompt)

                        logger.info(
                            "Rewrite LOCAL result=%s",
                            local_out
                        )
                        return local_out.strip()


            self.rewrite_llm = RewriteWithFallback()
            logger.info("Rewrite LLM 使用策略：FastAPI 远程优先 + 本地 fallback")

        except Exception as e:
            logger.warning(f"FastAPI Rewrite LLM 初始化失败，使用本地模型: {e}")
            self.rewrite_llm = local_llm

    def _load_and_index_documents(self):
        # 假设 self.collection_name = "medical_db"
        logger.debug(
            "Qdrant class location=%s, module=%s",
            inspect.getfile(Qdrant),
            Qdrant.__module__,
        )
        # 1. 检查数据文件
        logger.info("加载医学文档...")
        documents = []
        try:
            for file in os.listdir(self.data_path):
                # 检查是否为文件，避免尝试加载目录
                full_path = os.path.join(self.data_path, file)
                if file.endswith(".txt") and os.path.isfile(full_path):
                    logger.debug(f"加载文件: {file}")
                    # 使用 TextLoader，指定编码
                    loader = TextLoader(full_path, encoding="utf-8")
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = file
                    documents.extend(docs)
                else:
                    logger.debug(f"跳过非 .txt 文件或目录: {file}")
        except FileNotFoundError:
            logger.error(f"数据路径不存在: {self.data_path}")
            raise ValueError(f"数据路径不存在: {self.data_path}")
        except Exception as e:
            logger.error(f"加载文档时发生错误: {e}")
            raise

        if not documents:
            raise ValueError("未找到任何医学文档！请检查 data/ 目录并确保存在 .txt 文件。")

        # 2. 分块
        splitter = UnifiedMedicalTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        logger.info(f"成功加载 {len(documents)} 个文档，共生成 {len(chunks)} 个文本块，正在构建向量库...")
        # 优化后的打印逻辑
        for i, doc in enumerate(chunks):
            # 1. 为了防止日志太长，只截取前 100 个字符
            # 2. 去掉换行符，保持日志整洁
            preview_content = doc.page_content[:100].replace('\n', ' ')

            # 3. 获取元数据（方便检查 section 提取是否正确）
            source = doc.metadata.get("section", "未知章节")

            # 4. 使用 f-string 正确打印
            logger.debug(f"Chunk [{i}] | 章节: {source} | 内容: {preview_content}...")

        # 3. 构建/写入 Qdrant 向量库
        try:
            self.vector_store = Qdrant.from_documents(
                chunks,
                self.embedding_model,
                path="./qdrant_db",
                collection_name=self.collection_name,
                # 保持 force_recreate=True 以确保每次都是全新的索引
                # 如果写入失败，请检查文件系统权限
                force_recreate=False
            )
            logger.info("Qdrant 向量库构建完成。")
        except Exception as e:
            logger.error(f"Qdrant 写入失败，请检查 `./qdrant_db` 目录权限或磁盘空间。错误: {e}")
            raise

    def _graph_to_text(self, graph_record: dict) -> str:
        """
        【关键逻辑】将结构化图数据转换为自然语言文本
        """
        return (
            f"【知识图谱确证】药物<{graph_record['head']}>与<{graph_record['tail']}>"
            f"存在[{graph_record['relation']}]关系。\n"
            f"详细机制：{graph_record.get('description', '无详细描述')}。\n"
            f"数据来源：{graph_record.get('source', 'GraphDB')}"
        )

        # =========================================================================
        #  核心逻辑重写区域：双路检索 + 统一重排 + 拆解修复
        # =========================================================================

    def _hybrid_retrieve(self, query: str) -> List[Document]:
            """
            【架构核心】执行双路检索与统一重排 (Hybrid Retrieval & Rerank)

            逻辑流程：
            1. Path A (快路): 图数据库/规则检索 -> 转化为高置信度文本 -> Document
            2. Path B (慢路): 向量数据库检索 -> 获取上下文片段 -> Document
            3. Merge: 合并两路结果
            4. Rerank: 使用 Cross-Encoder 对所有结果进行语义相关性排序
            """
            logger.info(f"--- 开始双路检索: {query} ---")
            all_candidates: List[Document] = []

            # --- Path A: 快路 - 图引擎 (Rule/Graph Engine) ---
            if self.graph_retriever:
                try:
                    # GraphRetriever 内部已封装 NER + Cypher 查询
                    # 它返回的 Document 已经包含 "【知识图谱确证】" 这样的强语义前缀
                    graph_docs = self.graph_retriever.retrieve(query)
                    for doc in graph_docs:
                        # 标记来源类型，Metadata 可用于后续过滤或 UI 展示
                        doc.metadata["source_type"] = "KnowledgeGraph"
                        # 图谱结果通常是硬规则，给予默认高分元数据（虽然 Cross-Encoder 会重新打分）
                        doc.metadata["confidence"] = "high"

                    all_candidates.extend(graph_docs)
                    logger.info(f"Path A (Graph) 命中 {len(graph_docs)} 条结构化证据")
                except Exception as e:
                    logger.warning(f"图引擎检索异常: {e}")

            # --- Path B: 慢路 - RAG引擎 (Vector DB) ---
            if hasattr(self, 'vector_store'):
                try:
                    # 召回 Top-10，给 Reranker 足够的选择空间
                    # vector_docs = self.vector_store.similarity_search(query, k=10)
                    # for doc in vector_docs:
                    #     doc.metadata["source_type"] = "VectorDB"
                    #     doc.metadata["confidence"] = "medium"
                    # 使用 similarity_search_with_score 而不是 similarity_search
                    results_with_score = self.vector_store.similarity_search_with_score(query, k=5)

                    valid_docs = []
                    THRESHOLD = 0.65  # 需要根据你的 Embedding 模型测试确定，BGE-M3 通常建议 0.5~0.6

                    for doc, score in results_with_score:
                        logger.debug(f"检索得分: {score} | 内容: {doc.page_content[:20]}...")
                        if score >= THRESHOLD:
                            valid_docs.append(doc)
                        else:
                            logger.warning(f"丢弃低分文档 (Score: {score}): {doc.page_content[:20]}...")

                    if not valid_docs:
                        logger.info("未检索到符合阈值的文档。")

                    all_candidates.extend(valid_docs)
                    logger.info(f"Path B (Vector) 召回 {len(valid_docs)} 条非结构化片段")
                except Exception as e:
                    logger.error(f"向量检索异常: {e}")

            # --- Path C: 统一重排 (Unified Reranking) ---
            if not all_candidates:
                logger.warning("双路检索均无结果")
                return []

            # 1. 简单去重 (基于内容哈希)
            unique_docs = {}
            for doc in all_candidates:
                # 移除空白字符后做Hash
                doc_hash = hash(doc.page_content.replace(" ", "").strip())
                if doc_hash not in unique_docs:
                    unique_docs[doc_hash] = doc
            candidates_list = list(unique_docs.values())

            # 2. Cross-Encoder 重排
            try:
                logger.debug(f"合并后候选集 {len(candidates_list)} 条，开始 Rerank...")
                compressor = CrossEncoderReranker(model=self.reranker, top_n=3)
                reranked_docs = compressor.compress_documents(
                    documents=candidates_list,
                    query=query
                )

                # 日志记录最终选中的 Top-3
                for i, d in enumerate(reranked_docs):
                    src = d.metadata.get('source_type', 'Unknown')
                    logger.debug(f"Top-{i + 1} [{src}]: {d.page_content[:50]}...")

                return reranked_docs

            except Exception as e:
                logger.error(f"Rerank 失败，降级为原始顺序: {e}")
                return candidates_list[:3]

    # def _hybrid_retrieve2(self, query: str) -> List:
    #     """混合检索：向量 + 图 → 合并 → 统一 rerank"""
    #     # 1. 向量检索（原始相似性）
    #     logger.info(f"混合检索接收到查询： {query}")
    #     vector_docs = self.vector_store.similarity_search(query, k=5)
    #
    #     # 2. 图检索（如有）
    #     graph_docs = []
    #     if self.graph_retriever:
    #         try:
    #             graph_docs = self.graph_retriever.retrieve(query)
    #             for doc in graph_docs:
    #                 logger.info(f"  图检索 : {doc.page_content}")
    #         except Exception as e:
    #             logger.error(f"图检索异常（已跳过）: {e}")
    #
    #     # 3. 合并 & 去重
    #     all_docs = vector_docs + graph_docs
    #     seen = set()
    #     unique_docs = []
    #     for doc in all_docs:
    #         key = doc.page_content.strip()
    #         if key and key not in seen:
    #             seen.add(key)
    #             unique_docs.append(doc)
    #
    #     if not unique_docs:
    #         return []
    #     # 4. 统一 rerank（使用你原有的 reranker）
    #     compressor = CrossEncoderReranker(model=self.reranker, top_n=3)
    #     reranked_docs = compressor.compress_documents(documents=unique_docs, query=query)
    #     return reranked_docs

    def _rewrite_query(self, query: str) -> str:
        rewrite_rules = {
            "血糖高": "高血糖",
            "糖尿": "糖尿病",
            "怀孕血糖": "妊娠期糖尿病"
        }
        rewritten = query
        for k, v in rewrite_rules.items():
            rewritten = rewritten.replace(k, v)
        if rewritten != query:
            logger.debug(f"原始查询: {query} → 重写后: {rewritten}")
        return rewritten

    def _build_rag_chain(self):
        def format_docs_with_sources(docs):
            if not docs or all(not doc.page_content.strip() for doc in docs):
                return {"context": "", "sources": ""}
            context = "\n\n".join(doc.page_content for doc in docs)
            sources = sorted(set(doc.metadata.get("source", "未知来源") for doc in docs))
            return {"context": context, "sources": "、".join(sources)}

        prompt_template = """
你是一名专业医疗助手，请严格依据以下医学资料回答问题。
你的唯一信息来源是检索到的医学资料（Context）。请严格遵循以下规则回答。

【医学资料】
{context}

【用户问题】
{question}

【回答规则】
1. **只允许使用 Context 中出现的信息回答**；不得依据常识、不得凭经验补充、不得引用外部知识。
2. 若 Context 未包含答案，必须回复：  
   “根据当前知识库无法回答该问题。”
3. 你的回答必须：
   - 语言专业、准确、客观  
   - 内容简洁、结构清晰  
   - 不包含推测、杜撰、过度概括或专业建议外延
4. 不提供诊断、治疗方案、用药推荐、风险评估等医疗决策性内容。
5. 不输出任何可能被理解为医疗建议的语句，可以描述事实，但不能推论。
6. 回答格式：
   - 先给出“直接答案”（若能回答）  
   - 若相关知识在 Context 多处出现，可列出简要来源摘要（非 URL）

【输出格式】
请严格使用以下结构：

【回答】
（你的主要回答）

【依据】
（简要列出来自 Context 的依据，不生成新内容）
"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        query_rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一个医学查询规范化助手。请将用户的医学问题改写成一个标准、简洁、可用于检索的单句问句。\n"
             "要求：\n"
             "- 仅输出改写后的问句，且必须是完整的一句话。\n"
             "- 禁止输出任何解释、问候、追问、建议或额外文字。\n"
             "- 禁止输出 Markdown、引号、序号、前缀（如 'AI:'、'答：' 等）。\n"
             "- 改写后立即停止生成，不要继续说话。\n"
             "- 示例中的 AI 回答就是你输出的唯一格式。\n"
             "- 如果输入已是规范表达，直接原样输出，不做任何修改。\n"
             "- 输出必须是单行文本，以问号结尾（如原问题有问号）或保持陈述句（如原问题无问号）。\n"
             "- 绝对不要开启多轮对话，不要复述指令，不要自我确认。\n"
             ),
            ("human", "头痛怎么办？"),
            ("ai", "头痛的治疗方法是什么？"),
            ("human", "我肚子疼，是不是胃炎？"),
            ("ai", "腹痛是否由胃炎引起？"),
            ("human", "高血压吃啥口药好？"),
            ("ai", "高血压的推荐药物有哪些？"),
            ("human", "{query}")
        ])

        query_rewriter_chain = (
                query_rewrite_prompt #query_rewrite_prompt.format_prompt(query=rule_rewritten)
                | self.rewrite_llm
                | StrOutputParser()
        )

        def log_and_rewrite(question_str):
            logger.info(f"查询重写流程: 原始 → 规则 → LLM")
            logger.info("Rewrite input original_q=%s", question_str)
            # 1. 先用规则重写
            rule_rewritten = self._rewrite_query(question_str)
            logger.info(
                "Rewrite after rule, rule_rewritten_q=%s",
                rule_rewritten
            )
            # 2. 再用 LLM 规范化
            final_rewritten = query_rewriter_chain.invoke({"query": rule_rewritten})
            logger.info(
                "Rewrite final result, rewritten_q=%s",
                final_rewritten
            )
            return {"rewritten_q": final_rewritten, "original_q": question_str}

        input_mapper = RunnableLambda(log_and_rewrite)

        def retrieve_and_log(x):
            docs = self._hybrid_retrieve(x["rewritten_q"])  # 混合检索点
            logger.debug("\n混合检索后返回的文本块（Top 3）:")
            for i, doc in enumerate(docs, 1):
                content = doc.page_content.strip()[:200].replace('\n', ' ')
                source = doc.metadata.get("source", "未知")
                logger.debug(f"  [{i}] 来源: {source} | 内容: {content}...")
            return {"docs": docs, "question": x["original_q"]}

        def prepare_prompt_input(x):
            info = format_docs_with_sources(x["docs"])
            context = info["context"]
            sources = info["sources"]
            question = x["question"]
            logger.debug("最终提示词（Prompt）输入内容:")
            final_prompt = prompt.format(context=context, question=question)
            logger.debug(final_prompt)
            logger.debug("\n" + "=" * 60 + "\n")
            return {
                "context": context,
                "sources": sources,
                "question": question
            }

        base_chain = (
                input_mapper #查询重写
                | RunnableLambda(retrieve_and_log) #混合检索
                | RunnableLambda(prepare_prompt_input) #主查询
                | {
                    "answer": prompt | self.llm | StrOutputParser(),
                    "sources": lambda x: x["sources"],
                }
        )

        def finalize_output(inputs):
            answer = inputs["answer"]
            sources = inputs["sources"]
            if not sources or "根据当前知识库无法回答" in answer:
                return answer
            return f"{answer}\n\n（资料来源：{sources}）"

        self.rag_chain = base_chain | RunnableLambda(finalize_output)

    def _build_review_prompt(self, medical_text: str) -> str:
            return f"""
        请根据以下病历文本，严格按照给定的 JSON 结构进行信息抽取。
        【JSON 结构】
        {{
          "patient_info": {{
            "age": "string | null",       // 提取原文提到的年龄（带单位，如'24个月'或'45岁'）
            "gender": "string | null",
            "vital_signs_extracted": {{   // 仅提取原文明确记录的体征数据
                "temperature": "string | null",
                "blood_pressure": "string | null",
                "heart_rate": "string | null"
            }}
          }},
          "medical_history": {{
            "chief_complaint": "string | null",  // 主诉
            "history_present_illness": "string | null", // 现病史
            "past_medical_history": "string | null",    // 既往史/过敏史
            "symptoms_list": ["string"]   // 原文提到的具体症状列表（原子化抽取）
          }},
          "diagnosis_info": {{
            "clinical_diagnosis": ["string"], // 医生下达的诊断结果
            "icd_code_candidate": "string | null" // 若原文提到了ICD编码则提取，否则null
          }},
          "examinations": [ // 检查检验
            {{
              "name": "string", // 项目名称，如"血常规"、"CT"
              "findings": "string", // 检查所见/结果描述
              "is_abnormal": boolean // 仅根据原文描述判断（原文说异常即为true，否则false）
            }}
          ],
          "treatment_plan": {{ // 【重点：处置意见】
            "medications": [ // 处方/用药信息
              {{
                "name": "string",
                "specification": "string | null", // 规格/剂量
                "usage": "string | null" // 用法用量
              }}
            ],
            "procedures": ["string"], // 治疗操作（如：清创缝合、手术、吸氧）
            "disposition": "string | null", // 处置去向（如：离院、留观、收住院、转院）
            "doctor_advice": "string | null", // 医嘱/健康指导（如：低脂饮食、卧床休息、3天后复查）
            "follow_up_plan": "string | null" // 复诊计划
          }}
        }}
    
        【待处理病历文本】
        {medical_text}
        """.strip()

    def _invoke_llm(self, prompt: str) -> str:
        """
        调用 Ollama /api/chat，使用 system + user messages。
        prompt: 这里是最终提示词字符串（包含病历文本等）
        """
        logger.debug("调用 LLM（通过 OllamaClient /api/chat）")

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个专业的【医学病历结构化专员】"
                    "你的唯一任务是：从非结构化的病历文本中精准提取信息，填入指定的 JSON 字段（非诊断）。"
                    "【强制要求】 "
                    "- 只输出 JSON，不要输出任何解释性文字 - 不确定的信息请使用 null "
                    "- 不允许编造病历中未出现的信息 "
                    "- JSON 必须是合法格式，可被直接解析"
                    "- 不要对病情进行风险评估，不要给出你的医学建议，只提取原文记录的内容 "
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        return self.ollama.generate(messages)

    def _extract_json_block(self, text: str) -> str | None:
        match = re.search(r"\{[\s\S]*\}", text)
        return match.group(0) if match else None

    def _repair_json_with_llm(self, bad_json: str, medical_text: str) -> str:
        repair_prompt = f"""
    你是一个 JSON 修复助手。
    以下内容是一个【不合法的 JSON】，来源于医学病历检阅任务。

    【要求】
    - 只输出修复后的 JSON
    - 不要新增任何病历中不存在的信息
    - 保持原有 JSON 结构
    - 无法确定的字段设为 null

    【原始病历文本】
    {medical_text}

    【损坏的 JSON】
    {bad_json}
    """.strip()

        logger.info("尝试使用 LLM 修复 JSON")
        return self._safe_llm_call(repair_prompt, retry=1)

    def _parse_or_repair_json(self, text: str, medical_text: str) -> dict:
        """
        解析 JSON：
        1. 直接解析
        2. 提取 JSON 子串
        3. LLM 修复
        4. 最终降级
        """

        # 直接 parse
        try:
            return json.loads(text)
        except Exception:
            logger.warning("JSON 直接解析失败，尝试修复")

        # 尝试提取 {...}
        extracted = self._extract_json_block(text)
        if extracted:
            try:
                return json.loads(extracted)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"提取 JSON 后解析失败: {e.msg} at line {e.lineno}, col {e.colno}. 原始内容片段: {extracted[:200]!r}")
            except Exception as e:
                logger.warning(
                    f"提取 JSON 后发生未知解析错误: {type(e).__name__}: {e}. 原始内容片段: {extracted[:200]!r}")
        # 让 LLM 修复 JSON
        repaired = self._repair_json_with_llm(text, medical_text)
        try:
            return json.loads(repaired)
        except Exception:
            logger.error("LLM 修复 JSON 仍失败，进入降级")

        # 最终降级（返回最小安全结构）
        return self._fallback_empty_review()

    def _fallback_empty_review(self) -> dict:
        logger.error("进入病历检阅最终降级路径")
        return {
            "patient_summary": {
                "age": None,
                "gender": None,
                "chief_complaint": None,
                "history": None
            },
            "key_findings": [],
            "risk_flags": [],
            "medications": [],
            "tests": [],
            "review_conclusion": "病历信息不足，无法完成结构化检阅。"
        }

    def _execute_batch_audit(self, queries: List[str], case_context: dict) -> dict:
        """
        [修改版] 执行批量审核与整体总结
        流程：遍历原子查询 -> 混合检索 -> 单点验证 -> 汇总生成整体报告
        返回结构：
        {
            "details": [ ...单点审核结果... ],
            "overall_analysis": {
                "risk_level": "高/中/低",
                "action": "通过/拦截/慎用",
                "summary": "...",
                "suggestions": "..."
            }
        }
        """
        results = []

        # --- 阶段 1: 单点审核 (Map) ---
        for query in queries:
            try:
                # 1. 复用混合检索
                docs = self._hybrid_retrieve(query)
                if not docs:
                    continue

                # 2. 构造单点审核 Prompt
                context_text = "\n".join([d.page_content[:300] for d in docs])
                sources = list(set([d.metadata.get("source", "未知") for d in docs]))

                audit_prompt = f"""
                你是一名药品安全审核员。请依据证据对当前查询进行风险评估。
                【当前查询】: {query}
                【医学证据】: {context_text}
                【任务】: 判断是否存在用药风险。
                【输出要求】: 简练的一句话结论，指明风险等级(高/中/低/无)。
                """

                # 3. 调用 LLM (单点判断)
                review_res = self.ollama.generate([
                    {"role": "system", "content": "你是一个严谨的药学审核助手。请简练回答。"},
                    {"role": "user", "content": audit_prompt}
                ])

                results.append({
                    "query": query,
                    "evidence_sources": sources,
                    "ai_review": review_res
                })

            except Exception as e:
                logger.error(f"审核查询 '{query}' 时发生错误: {e}")

        # --- 阶段 2: 整体总结 (Reduce) ---
        if not results:
            return {
                "details": [],
                "overall_analysis": "未触发任何审核规则，无风险数据。"
            }

        logger.info("单点审核完成，正在生成整体综述报告...")
        try:
            # 1. 准备汇总上下文
            # 提取患者关键信息
            patient_info = case_context.get("patient_info", {})
            diagnosis_str = ", ".join(case_context.get("diagnosis_info", {}).get("clinical_diagnosis", []))

            # 将所有单点审核结果拼接成文本
            audit_trace = "\n".join([
                f"- 检查点: {r['query']}\n  AI发现: {r['ai_review']}"
                for r in results
            ])

            # 2. 构造“主审药师” Prompt
            summary_prompt = f"""
            你是一名三甲医院的【主任药师】。请根据下方的【患者信息】和【单项审核记录】，生成一份最终的用药安全综合评估报告。

            【患者信息】
            年龄: {patient_info.get('age', '未知')}, 性别: {patient_info.get('gender', '未知')}
            诊断: {diagnosis_str}

            【系统单项审核记录】
            {audit_trace}

            【你的任务】
            1. 综合分析所有检查结果，判断是否存在冲突（例如：一个通过，另一个提示高风险）。
            2. 给出最终的决策建议（通过 / 拦截 / 提示医生慎用）。
            3. 如果有风险，请按照严重程度排序说明。

            【输出格式 (JSON)】
            请直接输出合法的 JSON，不要包含 Markdown 标记：
            {{
                "final_decision": "通过/拦截/人工复核",
                "max_risk_level": "高/中/低/无",
                "summary_text": "简短的综合评价（100字以内）",
                "actionable_advice": "给医生的具体建议（如：建议停用XX药，改用XX）"
            }}
            """

            # 3. 调用 LLM (综述)
            final_verdict_raw = self.ollama.generate([
                {"role": "system", "content": "你是由系统生成的最终决策层，必须输出 JSON 格式。"},
                {"role": "user", "content": summary_prompt}
            ])

            # 4. 解析综述结果
            import json, re
            try:
                # 尝试提取 JSON
                match = re.search(r"\{[\s\S]*\}", final_verdict_raw)
                if match:
                    overall_analysis = json.loads(match.group(0))
                else:
                    overall_analysis = {"raw_text": final_verdict_raw}
            except:
                overall_analysis = {"raw_text": final_verdict_raw, "parse_error": "JSON解析失败"}

        except Exception as e:
            logger.error(f"生成整体总结时出错: {e}")
            overall_analysis = {"error": str(e)}

        return {
            "details": results,
            "overall_analysis": overall_analysis
        }

    def _decompose_case_to_atomic_queries(self, case_json: dict) -> List[str]:
        """
        【已修复】将病例拆解为原子查询，增加空值安全处理 (Null Safety)
        """
        # 1. 安全提取数据，防止 None 导致 crash
        patient = case_json.get("patient_info", {}) or {}
        diagnosis_info = case_json.get("diagnosis_info", {}) or {}
        clinical_diagnosis = diagnosis_info.get("clinical_diagnosis", []) or []
        treatment_plan = case_json.get("treatment_plan") or {}
        meds = treatment_plan.get("medications") or []

        # 2. 健壮的处方字符串构建
        real_drug_names = []  # 用于Prompt提示
        valid_meds_str_list = []
        for m in meds:
            # 只有当 m 是字典且 key 存在时才处理
            if isinstance(m, dict):
                name = m.get('name')
                usage = m.get('usage')
                # 只有当药名存在(非None非空)时才拼接
                if name:
                    real_drug_names.append(name)
                    med_str = f"{name} {usage or ''}".strip()
                    valid_meds_str_list.append(med_str)

        meds_context = ", ".join(valid_meds_str_list) if valid_meds_str_list else "无明确处方记录"
        drug_names_str = ", ".join(real_drug_names) if real_drug_names else "无"
        # 3. 构造 Prompt 上下文
        context_str = (
            f"患者: {patient.get('age', '未知')}, {patient.get('gender', '未知')}\n"
            f"诊断: {', '.join(clinical_diagnosis)}\n"
            f"处方: {meds_context}"
        )
        # 定义专门的 System Prompt (覆盖默认的结构化专员设定)
        system_instruction = (
            "你是一个专业的【医学检索查询生成器】。\n"
            "你的任务是生成搜索语句（String），而不是提取数据。\n"
            "【严禁】输出 Key-Value 字典或对象。\n"
            "【必须】输出纯字符串列表，例如：[\"查询语句1\", \"查询语句2\"]。"
        )
        # 定义 User Prompt (增强格式约束)
        user_prompt = f"""
                请将以下病例拆解为用于向量检索的【原子查询语句】。
 
        【病例信息】
        {context_str}
        
        【当前药物列表】
        {meds_context}

        【生成任务】(请严格执行，不要输出对象，只输出句子)
        1. 用法用量：
           - 生成：{drug_names_str}在{patient.get('age', '该年龄段')}中的用法用量
        2. 禁忌症：
           - 生成：{drug_names_str}的禁忌症
        3. 适应症匹配：
           - 对每个诊断生成：{drug_names_str}是否适用于[诊断]
        4. 相互作用：
           - (如果处方只有1种药，请忽略此项，不要生成null)
        
        【格式强制要求】
        - 输出必须是 JSON 字符串列表 (List[str])。
        - 列表中的元素必须是完整的自然语言句子。
        - 正确格式：["乳果糖的用法用量", "乳果糖是否适用于便秘"]

        【请直接输出结果，不要包含Markdown标记】
        """

        try:
            logger.debug("正在生成原子查询...")
            # *** 关键修改：直接构造 messages，不经过 _invoke_llm ***
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ]

            # 直接调用 ollama，绕过原有逻辑
            response = self.ollama.generate(messages)

            logger.debug("self._invoke_llm  返回%s ", response)
            # 清洗结果
            match = re.search(r"\[[\s\S]*\]", response)
            if match:
                queries = json.loads(match.group(0))
                return [str(q) for q in queries if isinstance(q, str)]
            else:
                logger.warning("查询拆解未能解析出 JSON 列表，使用后备规则。")
                return self._fallback_decomposition(meds)
        except Exception as e:
            logger.error(f"查询拆解失败: {e}")
            return self._fallback_decomposition(meds)


    def _fallback_decomposition(self, meds: List[dict]) -> List[str]:
        """降级策略：基于规则的简单拆解"""
        queries = []
        for m in meds:
            name = m.get('name')
            if name:
                queries.append(f"{name} 说明书 用法用量")
                queries.append(f"{name} 禁忌症")
                queries.append(f"{name} 药物相互作用")
        return queries

    def _safe_llm_call(self, prompt: str, retry: int = 2) -> str:
        last_error = None

        for attempt in range(1, retry + 1):
            try:
                logger.info("LLM 调用尝试 #%d", attempt)
                output = self._invoke_llm(prompt)
                logger.debug("LLM 原始输出：%s", output)
                return output
            except Exception as e:
                last_error = e
                logger.warning("LLM 调用失败 #%d: %s", attempt, e)

        raise RuntimeError("LLM 多次调用失败") from last_error

    def review_record(self, medical_text: str) -> dict:
        """
        病历检阅主入口：
        - 返回 dict（已解析 JSON）
        - 内部完成 retry / repair / fallback
        """
        logger.info("开始病历检阅（review_record）")
        prompt = self._build_review_prompt(medical_text)
        # 第一次尝试
        raw_output = self._safe_llm_call(prompt)
        # JSON 解析 + 自动修复
        extracted_data = self._parse_or_repair_json(raw_output, medical_text)
        logger.info("病历检阅完成（JSON 已结构化）%s", extracted_data)
        # 如果抽取失败或为空，直接返回
        if not extracted_data.get("treatment_plan", {}).get("medications"):
            logger.warning("未检测到处方信息，跳过用药审核步骤。")
            extracted_data["audit_report"] = "无处方信息，无法审核。"
            return extracted_data
        # --- 步骤 2: 原子化查询拆解 (Requirement 2) ---
        # 使用本地 LLM 将结构化数据拆解为多个具体的检索 Query
        logger.info("正在进行原子化查询拆解...")
        atomic_queries = self._decompose_case_to_atomic_queries(extracted_data)

        # --- 步骤 3: RAG 辅助审核 ---
        # 针对每个原子查询进行 RAG 检索和风险判断
        logger.debug(f"生成的原子查询列表: {atomic_queries}")
        # audit_results = self._execute_batch_audit(atomic_queries, extracted_data)
        # 执行审核，现在返回的是包含 summary 的字典
        full_audit_result = self._execute_batch_audit(atomic_queries, extracted_data)

        # --- 步骤 4: 结果注入 ---
        # 将结果存入 extracted_data
        # 建议把 details 和 summary 分开存，方便前端展示
        extracted_data["audit_report_details"] = full_audit_result["details"]
        extracted_data["audit_report_summary"] = full_audit_result["overall_analysis"]

        # 为了兼容旧逻辑，可以保留 audit_report 字段
        extracted_data["audit_report"] = full_audit_result

        logger.info("病历检阅完成。")
        return extracted_data

    def ask(self, question: str) -> str:
        return self.rag_chain.invoke(question)