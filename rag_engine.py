# rag_engine.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用国内镜像
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# 禁用警告
logging.getLogger("langchain").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


class MedicalRAG:
    def __init__(self, data_path: str = "data/", collection_name: str = "medical_db"):
        self.data_path = data_path
        self.collection_name = collection_name
        self._init_models()
        self._load_and_index_documents()
        self._build_rag_chain()

    def _init_models(self):
        print("加载嵌入模型 (BAAI/bge-m3)...")
        try:
            # 尝试加载本地模型
            model_path = "models/bge-m3"
            if os.path.exists(model_path):
                print(f"→ 使用本地模型: {model_path}")
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=model_path,
                    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
            else:
                # 网络下载（备用）
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-m3",
                    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
        except Exception as e:
            print(f"⚠️ 嵌入模型加载失败: {str(e)}")
            print("→ 使用轻量级备用模型: sentence-transformers/all-MiniLM-L6-v2")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )

        print("加载重排器 (BAAI/bge-reranker-v2-m3)...")
        try:
            reranker_path = "models/bge-reranker-v2-m3"
            if os.path.exists(reranker_path):
                print(f"→ 使用本地重排器模型: {reranker_path}")
                self.reranker = HuggingFaceCrossEncoder(model_name=reranker_path)
            else:
                self.reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        except Exception as e:
            print(f"⚠️ 重排器加载失败: {str(e)}")
            print("→ 使用备用重排器: cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.reranker = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

        print("加载本地 LLM (Qwen/Qwen2.5-0.5B-Instruct)...")
        try:
            llm_path = "models/qwen2.5-0.5b"
            if os.path.exists(llm_path):
                print(f"→ 使用本地LLM: {llm_path}")
                model = AutoModelForCausalLM.from_pretrained(
                    llm_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    trust_remote_code=True
                )
                tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
            else:
                model_name = "Qwen/Qwen2.5-0.5B-Instruct"
                print(f"→ 从Hugging Face下载LLM: {model_name} (这可能需要几分钟)")
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    trust_remote_code=True
                )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=False,
                repetition_penalty=1.1
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            print(f"⚠️ LLM加载失败: {str(e)}")
            print("→ 使用超轻量级模型: microsoft/Phi-3-mini-4k-instruct (需要在线下载)")
            try:
                from transformers import Phi3ForCausalLM, AutoTokenizer
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
                print(f"⚠️ 备用LLM加载失败: {str(e2)}")
                print("❌ 无法加载任何语言模型。请检查网络连接或创建 models/ 目录并放置本地模型。")
                raise

    def _load_and_index_documents(self):
        print("加载医学文档...")
        documents = []
        for file in os.listdir(self.data_path):
            if file.endswith(".txt"):
                loader = TextLoader(os.path.join(self.data_path, file), encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file
                documents.extend(docs)

        if not documents:
            raise ValueError("未找到任何医学文档！请检查 data/ 目录。")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "；", " "]
        )
        chunks = text_splitter.split_documents(documents)

        print(f"共生成 {len(chunks)} 个文本块，正在构建向量库...")
        self.vector_store = Qdrant.from_documents(
            chunks,
            self.embedding_model,
            location=":memory:",
            collection_name=self.collection_name
        )

        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        compressor = CrossEncoderReranker(model=self.reranker, top_n=3)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

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
            print(f"原始查询: {query} → 重写后: {rewritten}")
        return rewritten

    def _build_rag_chain(self):
        prompt_template = """
你是一名专业医疗助手，请严格依据以下医学资料回答问题。
如果资料中无相关信息，请回答："根据当前知识库无法回答该问题"。

【医学资料】
{context}

【问题】
{question}

【要求】
1. 回答必须准确、简洁；
2. 必须注明资料来源（如文件名或指南名称）；
3. 禁止任何猜测、编造或超出资料范围的建议。
"""
        prompt = ChatPromptTemplate.from_template(prompt_template)

        self.rag_chain = (
                {
                    "context": lambda x: self.retriever.invoke(self._rewrite_query(x["question"])),
                    "question": lambda x: x["question"]
                }
                | prompt
                | self.llm
                | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        return self.rag_chain.invoke({"question": question})