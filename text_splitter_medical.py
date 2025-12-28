import re
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


class UnifiedMedicalTextSplitter:
    def __init__(self, chunk_size=600, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", ";"]
        )

        # === 1. 增强正则匹配逻辑 ===

        # 药品说明书
        p_drug = r"(?:【[^】]+】|\[(?!\d+(?:-\d+)?\])[^\]]+\])"

        # 指南层级标题 (增强容错：允许前面没有换行符，只要后面跟着文字)
        p_guide_l1 = r"(?:^|\n|)\s*[一二三四五六七八九十]+、"
        p_guide_l2 = r"(?:^|\n|)\s*[（(][一二三四五六七八九十]+[）)]"
        p_guide_l3 = r"(?:^|\n|)\s*\d+\.\s"

        # *** 新增：针对本指南的特殊关键词 ***
        # 匹配 "临床问题 1：" 或 "推荐意见 8.2："
        p_clinical_q = r"(?:^|\n|)\s*临床问题\s*\d+[：:.]?"
        p_rec_opinion = r"(?:^|\n|)\s*推荐意见\s*[\d.]+[：:.]?"

        # 组合正则
        self.master_pattern = f"({p_drug}|{p_clinical_q}|{p_rec_opinion}|{p_guide_l1}|{p_guide_l2}|{p_guide_l3})"

    def _clean_text(self, text: str) -> str:
        """清洗与格式化"""
        # 1. 暴力清洗 PDF 的页眉页脚干扰 (如日志中的文件名)
        text = re.sub(r"来源：.*\.pdf", "", text)

        # 2. 去除文献引用 [12], [12-15]
        text = re.sub(r"\[\d+(?:[–-]\d+)?\]", "", text)

        # 3. 去除孤立页码
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

        # *** 关键修复：强制在标题前加换行符 ***
        # PDF 提取时经常把 "一段结束。二、下一段" 连在一起
        # 我们手动把 "二、" 前面加上换行，让 split 更容易识别

        # 处理 "二、xxx"
        text = re.sub(r"([。；;])\s*([一二三四五六七八九十]+、)", r"\1\n\2", text)
        # 处理 "临床问题"
        text = re.sub(r"([。；;])\s*(临床问题)", r"\1\n\2", text)
        # 处理 "推荐意见"
        text = re.sub(r"([。；;])\s*(推荐意见)", r"\1\n\2", text)

        # 4. 压缩空行
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def split_documents(self, documents: List[Document]) -> List[Document]:
        final_chunks = []

        for doc in documents:
            # 获取文件名作为强上下文（通常文件名就是药名，如 "左氧氟沙星片.txt"）
            source_name = doc.metadata.get("source", "").replace(".txt", "").replace(".pdf", "")
            # 1. 预清洗
            raw_text = self._clean_text(doc.page_content)

            # 2. 基于正则的一级结构化切分
            # segments 结果示例: ['', '二、...治疗', '内容...', '（一）...分类', '内容...']
            segments = re.split(self.master_pattern, raw_text)

            current_context = {
                "l1": "",  # 一、
                "l2": "",  # （一）
                "last_header": "前言"
            }

            # 处理开头可能存在的无标题前言
            if segments and not re.match(self.master_pattern, segments[0] or ""):
                preamble = segments[0].strip()
                if preamble:
                    final_chunks.extend(self._create_chunks(preamble, "前言", doc.metadata))
                segments = segments[1:]

            # 遍历 Header + Content 对
            for i in range(0, len(segments) - 1, 2):
                header = segments[i].strip()  # 如 "1. 手术评估"
                content = segments[i + 1].strip()

                if not content:
                    continue

                # --- 智能上下文追踪 (Context Tracking) ---
                # 目的：将父级标题拼接到子级内容中

                # 判断 Header 等级
                is_l1 = bool(re.match(r"^[一二三四五六七八九十]+、", header))
                is_l2 = bool(re.match(r"^[（(][一二三四五六七八九十]+[）)]", header))

                clean_header = re.sub(r"[【】\[\]]", "", header)

                if is_l1:
                    current_context["l1"] = clean_header
                    current_context["l2"] = ""  # 换章了，节清空
                elif is_l2:
                    current_context["l2"] = clean_header

                # 构建语义完整的块内容
                # 格式：[二、NSCLC治疗] [（一）可切除类] 1. 手术评估 \n 具体内容...
                # 这样即使切得很细，RAG 也能知道这是“NSCLC”的“手术评估”

                full_text = f"药品：{source_name}\n章节：{clean_header}\n内容：{content}"

                # 更新 Metadata
                new_metadata = doc.metadata.copy()
                new_metadata["section"] = clean_header
                new_metadata["parent_section"] = current_context["l1"]

                # 二级切分（长度控制）
                chunks = self._create_chunks(full_text, clean_header, new_metadata)
                final_chunks.extend(chunks)

        return final_chunks

    def _create_chunks(self, text: str, section_name: str, metadata: dict) -> List[Document]:
        """使用递归切分器处理长度"""
        return self.recursive_splitter.create_documents([text], metadatas=[metadata])