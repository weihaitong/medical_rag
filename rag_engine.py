# rag_engine.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import logging
from typing import List, Optional
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
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
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Phi3ForCausalLM,
    pipeline
)
# 配置日志器（模块级）
logger = logging.getLogger(__name__)
logger.info(f"HF_ENDPOINT set to: {os.environ.get('HF_ENDPOINT')}")
# 全局日志配置（仅在首次导入时生效）
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)  # 可根据需要调整为 logging.DEBUG

logger.propagate = False#防止打印两遍日志
# 禁用部分 noisy 日志（可选）
logging.getLogger("langchain").setLevel(logging.ERROR)
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
                    # === 关键：兼容 ChatPromptValue ===
                    if isinstance(input, ChatPromptValue):
                        # 取最后一条 HumanMessage
                        messages = input.to_messages()
                        human_msgs = [m for m in messages if m.type == "human"]
                        query = human_msgs[-1].content if human_msgs else messages[-1].content
                    else:
                        query = str(input)

                    try:
                        start = time.time()
                        result = remote_llm.invoke(query)
                        logger.info(f"Rewrite remote latency={time.time() - start:.2f}s")
                        return result.strip().splitlines()[0]
                    except Exception as e:
                        logger.info(f"Rewrite LLM 降级为本地模型，原因: {e}")
                        return local_llm.invoke(query)

            self.rewrite_llm = RewriteWithFallback()
            logger.info("Rewrite LLM 使用策略：FastAPI 远程优先 + 本地 fallback")

        except Exception as e:
            logger.warning(f"FastAPI Rewrite LLM 初始化失败，使用本地模型: {e}")
            self.rewrite_llm = local_llm

    def _load_and_index_documents(self):
        # 假设 self.collection_name = "medical_db"
        import inspect
        print("Qdrant class location:", inspect.getfile(Qdrant))
        print("Qdrant module:", Qdrant.__module__)
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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "；", " "]
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"成功加载 {len(documents)} 个文档，共生成 {len(chunks)} 个文本块，正在构建向量库...")

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

        # 4. 配置检索器
        # base_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        # compressor = CrossEncoderReranker(model=self.reranker, top_n=3)
        # self.retriever = ContextualCompressionRetriever(
        #     base_compressor=compressor,
        #     base_retriever=base_retriever
        # )
        # logger.info("向量检索器配置完成。")

    def _hybrid_retrieve(self, query: str) -> List:
        """混合检索：向量 + 图 → 合并 → 统一 rerank"""
        # 1. 向量检索（原始相似性）
        logger.info(f"混合检索接收到查询： {query}")
        vector_docs = self.vector_store.similarity_search(query, k=5)

        # 2. 图检索（如有）
        graph_docs = []
        if self.graph_retriever:
            try:
                graph_docs = self.graph_retriever.retrieve(query)
                for doc in graph_docs:
                    logger.info(f"  图检索 : {doc.page_content}")
            except Exception as e:
                logger.error(f"图检索异常（已跳过）: {e}")

        # 3. 合并 & 去重
        all_docs = vector_docs + graph_docs
        seen = set()
        unique_docs = []
        for doc in all_docs:
            key = doc.page_content.strip()
            if key and key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        if not unique_docs:
            return []
        # 4. 统一 rerank（使用你原有的 reranker）
        compressor = CrossEncoderReranker(model=self.reranker, top_n=3)
        reranked_docs = compressor.compress_documents(documents=unique_docs, query=query)
        return reranked_docs

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
                query_rewrite_prompt
                | self.rewrite_llm
                | StrOutputParser()
        )

        def log_and_rewrite(question_str):
            logger.debug(f"原始查询: {question_str}")
            # 1. 先用规则重写
            rule_rewritten = self._rewrite_query(question_str)
            # 2. 再用 LLM 规范化
            final_rewritten = query_rewriter_chain.invoke({"query": rule_rewritten})
            logger.info(f"查询重写流程: 原始 → 规则 → LLM")
            logger.info(f"  原始问题: {question_str}")
            logger.info(f"  规则重写: {rule_rewritten}")
            logger.info(f"  LLM重写: {final_rewritten}")
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

    def ask(self, question: str) -> str:
        return self.rag_chain.invoke(question)