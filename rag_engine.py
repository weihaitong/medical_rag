import logging
import json
import re
import ast
import os  # 【修正】确保导入 os
from typing import List, Optional

from langchain_core.documents import Document
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 引入解耦的模块
from model_factory import ModelFactory
from knowledge_manager import KnowledgeManager

# 【修正 1】引入图检索模块 (Read)
try:
    from graph_retriever import GraphRetriever

    GRAPH_RETRIEVER_AVAILABLE = True
except ImportError:
    GRAPH_RETRIEVER_AVAILABLE = False

# 【修正 2】引入图谱管理模块 (Write)
# 假设你把之前的写入代码保存为 graph_builder.py
try:
    from graph_builder import MedicalGraphManager

    GRAPH_WRITE_AVAILABLE = True
except ImportError:
    GRAPH_WRITE_AVAILABLE = False
    MedicalGraphManager = None

logger = logging.getLogger(__name__)


class MedicalRAG:
    """RAG 引擎：保留核心业务逻辑，协调 ModelFactory 和 KnowledgeManager"""

    def __init__(self, data_path: str = "data/", collection_name: str = "medical_db"):
        # 1. 引用模型工厂
        self.models = ModelFactory()

        # 2. 引用知识库管理 (向量库)
        self.knowledge = KnowledgeManager(
            data_path=data_path,
            embedding_model=self.models.embedding_model,
            collection_name=collection_name
        )

        # 3. 初始化图谱组件
        # 3.1 写组件 (Manager)
        self.graph_manager = None
        if GRAPH_WRITE_AVAILABLE:
            try:
                self.graph_manager = MedicalGraphManager()
            except Exception as e:
                logger.error(f"图谱管理器初始化失败: {e}")

        # 3.2 读组件 (Retriever)
        self.graph_retriever = GraphRetriever() if GRAPH_RETRIEVER_AVAILABLE else None

        # 4. 构建业务链
        self._build_rag_chain()

        # 动态配置参数 (默认值)
        self.retrieval_k = 20
        self.retrieval_threshold = 0.65
        self.enable_rerank = True
        self.rerank_top_n = 10

       # 快模型：用于结构化提取、查询拆解 (速度快，逻辑要求低)
        self.fast_model = "qwen3:1.7b"
        # 慢模型：用于最终审核、风险决策 (逻辑强，速度慢)
        self.smart_model = "qwen2:7b-instruct-q5_K_M"

    # =========================================================================
    #  【修正 3】将这些方法移出 __init__，作为类的方法
    # =========================================================================
    def add_knowledge_file(self, file_path: str):
        """
        全流程新增一个知识文件：
        1. 读取文本
        2. 存入向量库 (Vector DB)
        3. 存入图数据库 (Graph DB)
        """
        if not os.path.exists(file_path):
            return "文件不存在"

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        filename = os.path.basename(file_path)

        results = {}

        # 1. 向量库更新
        try:
            vec_success = self.knowledge.add_document(text, filename)
            results["vector_db"] = "Success" if vec_success else "Failed"
        except Exception as e:
            results["vector_db"] = f"Error: {e}"

        # 2. 图数据库更新
        if self.graph_manager:
            try:
                graph_success = self.graph_manager.add_document(text, filename)
                results["graph_db"] = "Success" if graph_success else "Failed"
            except Exception as e:
                results["graph_db"] = f"Error: {e}"
        else:
            results["graph_db"] = "Skipped (Manager unavailable)"

        return results

    def delete_drug_knowledge(self, drug_name: str):
        """
        从图谱中删除药品 (向量库删除通常较复杂，暂只演示图谱删除)
        """
        if self.graph_manager:
            res = self.graph_manager.delete_document(drug_name)
            return f"药品 [{drug_name}] 图谱节点已删除: {res}"
        return "图谱管理器未加载，无法删除。"

    # =========================================================================
    #  业务功能 1: 混合检索
    # =========================================================================
    def _hybrid_retrieve(self, query: str) -> List[Document]:
        """执行双路检索与统一重排 (Hybrid Retrieval & Rerank)"""
        logger.info(f"--- 开始双路检索: {query} ---")
        all_candidates: List[Document] = []

        # Path A: 图引擎 (Graph)
        if self.graph_retriever:
            try:
                # 调用 GraphRetriever.retrieve (返回 List[Document])
                graph_docs = self.graph_retriever.retrieve(query)

                for doc in graph_docs:
                    doc.metadata["source_type"] = "KnowledgeGraph"
                    # 给图谱确凿事实较高的初始权重
                    doc.metadata["confidence"] = "high"

                if graph_docs:
                    logger.info(f"Path A (Graph) 命中 {len(graph_docs)} 条结构化证据")
                    all_candidates.extend(graph_docs)
            except Exception as e:
                logger.warning(f"图引擎检索异常: {e}")

        # Path B: 向量引擎 (Vector)
        if self.knowledge.vector_store:
            try:
                results_with_score = self.knowledge.vector_store.similarity_search_with_score(query, k=self.retrieval_k)
                valid_docs = []
                current_threshold = self.retrieval_threshold
                for doc, score in results_with_score:
                    if score >= current_threshold:
                        doc.metadata["source_type"] = "VectorDB"
                        valid_docs.append(doc)
                logger.info(f"Path B (Vector) 召回 {len(valid_docs)} 条非结构化片段")
                all_candidates.extend(valid_docs)
            except Exception as e:
                logger.error(f"向量检索异常: {e}")

        # Path C: 统一重排 (Rerank)
        if not all_candidates:
            return []

        # 去重
        unique_docs = {}
        for doc in all_candidates:
            doc_hash = hash(doc.page_content.replace(" ", "").strip())
            if doc_hash not in unique_docs:
                unique_docs[doc_hash] = doc
        candidates_list = list(unique_docs.values())

        try:
            logger.debug(f"合并后候选集 {len(candidates_list)} 条，开始 Rerank...")
            # 这里的 rerank_top_n 使用 self.rerank_top_n 保持配置一致性
            compressor = CrossEncoderReranker(model=self.models.reranker, top_n=self.rerank_top_n)
            reranked_docs = compressor.compress_documents(documents=candidates_list, query=query)

            # 调试日志
            for i, d in enumerate(reranked_docs):
                src = d.metadata.get("source_type", "Unknown")  # 优化日志显示 source_type
                content_preview = d.page_content[:30].replace("\n", " ")
                logger.debug(f"Top-{i + 1} [{src}]: {content_preview}...")

            return reranked_docs
        except Exception as e:
            logger.error(f"Rerank 失败，降级为原始顺序: {e}")
            return candidates_list[:self.rerank_top_n]

    # ... (后续的 _rewrite_query, _build_rag_chain, ask 等方法不需要修改，保持你原代码即可)

    def _rewrite_query(self, query: str) -> str:
        # (保持你原代码不变)
        rewrite_rules = {
            "血糖高": "高血糖",
            "糖尿": "糖尿病",
            "怀孕血糖": "妊娠期糖尿病"
        }
        rewritten = query
        for k, v in rewrite_rules.items():
            rewritten = rewritten.replace(k, v)
        return rewritten

    def _build_rag_chain(self):
        # (保持你原代码不变)
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

        # Rewrite Chain
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
            ("human", "{query}")
        ])

        query_rewriter_chain = (
                query_rewrite_prompt
                | self.models.rewrite_llm
                | StrOutputParser()
        )

        def log_and_rewrite(question_str):
            logger.info(f"查询重写流程: 原始 → 规则 → LLM")
            rule_rewritten = self._rewrite_query(question_str)
            final_rewritten = query_rewriter_chain.invoke({"query": rule_rewritten})
            logger.info(f"Rewrite: {question_str} -> {rule_rewritten} -> {final_rewritten}")
            return {"rewritten_q": final_rewritten, "original_q": question_str}

        def retrieve_and_log(x):
            docs = self._hybrid_retrieve(x["rewritten_q"])
            return {"docs": docs, "question": x["original_q"]}

        def prepare_prompt_input(x):
            info = format_docs_with_sources(x["docs"])
            return {
                "context": info["context"],
                "sources": info["sources"],
                "question": x["question"]
            }

        self.rag_chain = (
                RunnableLambda(log_and_rewrite)
                | RunnableLambda(retrieve_and_log)
                | RunnableLambda(prepare_prompt_input)
                | {
                    "answer": prompt | self.models.llm | StrOutputParser(),
                    "sources": lambda x: x["sources"],
                }
        )

    def ask(self, question: str) -> str:
        """问答模式入口"""
        result = self.rag_chain.invoke(question)
        answer = result["answer"]
        sources = result["sources"]
        if not sources or "根据当前知识库无法回答" in answer:
            return answer
        return f"{answer}\n\n（资料来源：{sources}）"

    # =========================================================================
    #  业务功能 2: 病历检阅 (Review Record)
    #  (以下部分保持你原代码不变，不需要修改)
    # =========================================================================
    def _build_review_prompt(self, medical_text: str) -> str:
        return f"""
        你是一名资深的病历结构化专家。请阅读下方的【待处理病历文本】，提取关键信息填入指定的 JSON 格式。

        【提取规则】
        1. **完整性**：如果处方中有多种药物，**必须全部提取**，存入 medications 列表，严禁遗漏。
        2. **准确性**：数值和单位必须与原文一致。
        3. **格式**：只输出标准的 JSON 字符串，不要包含 Markdown 标记（如 ```json），不要包含注释。
        4. **极简输出**：仅输出病历中明确提及的字段。未提及的字段**不要包含在 JSON 中**，直接省略。
        5. **兼容性**：必须输出标准 JSON (null, true, false)，严禁使用 Python 风格 (None, True, False)。
        【目标 JSON 结构定义】
        - patient_info: 患者基本信息 (age, gender, vital_signs_extracted)
        - medical_history: 病史 (chief_complaint, symptoms_list=[症状1, 症状2...])
        - diagnosis_info: 诊断 (clinical_diagnosis=[诊断1, 诊断2...])
        - treatment_plan: 治疗方案 (medications=[{{name, specification, usage}}])

        【输出模板】
        {{
          "patient_info": {{
            "age": "string | null",
            "gender": "string | null",
            "vital_signs_extracted": {{
                "temperature": "string | null",
                "blood_pressure": "string | null",
                "heart_rate": "string | null"
            }}
          }},
          "medical_history": {{
            "chief_complaint": "string | null",
            "history_present_illness": "string | null",
            "past_medical_history": "string | null",
            "symptoms_list": ["string"]
          }},
          "diagnosis_info": {{
            "clinical_diagnosis": ["string"],
            "icd_code_candidate": "string | null" 
          }},
          "examinations": [
            {{
              "name": "string",
              "findings": "string",
              "is_abnormal": boolean
            }}
          ],
          "treatment_plan": {{ 
            "medications": [
              {{
                "name": "string",
                "specification": "string | null",
                "usage": "string | null" 
              }}
            ],
            "procedures": ["string"],
            "disposition": "string | null",
            "doctor_advice": "string | null",
            "follow_up_plan": "string | null"
          }}
        }}

        【待处理病历文本】
        {medical_text}
        """.strip()

    def _invoke_llm(self, prompt: str) -> str:
        """调用 Ollama Client 生成"""
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个专业精准的【医学病历结构化专员】"
                    "你的唯一任务是：从非结构化的病历文本中精准提取信息，填入指定的 JSON 字段（非诊断）。"
                    "【强制要求】 "
                    "- 只输出 JSON格式，不要输出任何解释性文字或注释 - 不确定的信息请使用 null "
                    "- 不允许编造病历中未出现的信息 "
                    "- JSON 必须是合法格式，可被直接解析"
                    "- 不要对病情进行风险评估，不要给出你的医学建议，只提取原文记录的内容 "
                    "- 当病历中包含多个药物时，必须**全部列出**，绝不能只提取第一个"
                    "- 如果没有相关信息，请保留字段并赋值为 null。"
                )
            },
            {"role": "user", "content": prompt}
        ]
        return self.models.ollama.generate(messages, model=self.fast_model)

    def _safe_llm_call(self, prompt: str, retry: int = 2) -> str:
        last_error = None
        for attempt in range(1, retry + 1):
            try:
                return self._invoke_llm(prompt)# _invoke_llm 内部默认使用了 self.fast_model
            except Exception as e:
                last_error = e
                logger.warning(f"LLM 调用失败 # {attempt}: {e}")
        raise RuntimeError("LLM 多次调用失败") from last_error

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

    def _fallback_empty_review(self) -> dict:
        logger.error("进入病历检阅最终降级路径")
        return {
            "patient_summary": {"age": None, "gender": None, "chief_complaint": None, "history": None},
            "key_findings": [], "risk_flags": [], "medications": [], "tests": [],
            "review_conclusion": "病历信息不足，无法完成结构化检阅。"
        }

    def _extract_json_block(self, text: str) -> str | None:
        match = re.search(r"\{[\s\S]*\}", text)
        return match.group(0) if match else None

    def _parse_or_repair_json(self, text: str, medical_text: str) -> dict:
        """
        解析 JSON，具备极强的容错能力：
        1. 去除 Markdown
        2. 尝试标准 JSON 解析 (json.loads)
        3. 尝试 Python 字面量解析 (ast.literal_eval) -> 解决 None/True/False/单引号问题
        4. 尝试正则修正 (None->null)
        5. LLM 修复
        """
        # --- 预处理：去除 Markdown 标记 ---
        cleaned_text = text.strip()
        # 去除 ```json ... ``` 或 ``` ... ```
        if cleaned_text.startswith("```"):
            # 找到第一个换行符
            first_newline = cleaned_text.find("\n")
            if first_newline != -1:
                cleaned_text = cleaned_text[first_newline + 1:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        # --- 策略 1: 标准 JSON 解析 ---
        try:
            return json.loads(cleaned_text)
        except Exception:
            pass

            # --- 策略 2: Python 语法解析 (关键修复) ---
        # 模型经常输出 Python dict (None, True, False, 'string') 而不是 JSON
        try:
            # ast.literal_eval 能安全解析 Python 风格的字典字符串
            return ast.literal_eval(cleaned_text)
        except Exception:
            pass

        # --- 策略 3: 提取 JSON 块并重试上述步骤 ---
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            block = match.group(0)
            try:
                return json.loads(block)
            except:
                try:
                    return ast.literal_eval(block)
                except:
                    pass

        # --- 策略 4: 暴力正则替换 (最后的挣扎) ---
        # 如果 ast 失败了（可能是因为不合法的语法），尝试手动替换关键字
        try:
            # 替换 Python 关键字为 JSON 关键字
            # 使用 \b 确保是单词边界，避免替换掉单词内部的字符
            fixed_text = cleaned_text
            fixed_text = re.sub(r'\bNone\b', 'null', fixed_text)
            fixed_text = re.sub(r'\bTrue\b', 'true', fixed_text)
            fixed_text = re.sub(r'\bFalse\b', 'false', fixed_text)
            # 尝试将单引号替换为双引号 (风险较大，仅作尝试)
            # fixed_text = fixed_text.replace("'", '"')
            return json.loads(fixed_text)
        except Exception:
            pass

        # --- 策略 5: LLM 修复 ---
        logger.warning(f"JSON/AST 解析均失败，尝试 LLM 修复。片段: {cleaned_text[:50]}...")
        repaired = self._repair_json_with_llm(text, medical_text)
        try:
            # 修复后的结果也可能包含 Python 语法，所以再次尝试双重解析
            try:
                return json.loads(repaired)
            except:
                return ast.literal_eval(repaired)
        except Exception as e:
            logger.error(f"LLM 修复后仍无法解析: {e}")
            return self._fallback_empty_review()

    def _decompose_case_to_atomic_queries(self, case_json: dict) -> List[str]:
        """
        [全面优化版] 将病例拆解为原子查询

        优化策略：
        1. **维度扩展**：增加药物相互作用、特殊人群、重复用药等审核维度。
        2. **效率提升**：
           - 单药查询合并：将适应症、禁忌、用法合并为一条"说明书综合查询"。
           - 交互查询合并：将所有药物组合生成一条"相互作用查询"。
        3. **鲁棒性**：增强 JSON 解析能力。
        """
        import ast
        import json
        import re

        # 1. 数据提取与预处理
        patient = case_json.get("patient_info", {}) or {}
        diagnosis_info = case_json.get("diagnosis_info", {}) or {}
        clinical_diagnosis = diagnosis_info.get("clinical_diagnosis", []) or []
        diagnosis_str = ", ".join(clinical_diagnosis) if clinical_diagnosis else "未明确诊断"

        treatment_plan = case_json.get("treatment_plan") or {}
        meds = treatment_plan.get("medications") or []

        # 提取药物名列表
        drug_names = []
        for m in meds:
            if isinstance(m, dict) and m.get('name'):
                drug_names.append(m.get('name'))

        if not drug_names:
            logger.warning("未提取到药物名称，跳过拆解。")
            return []

        # 2. 患者特征泛化 (用于增强检索相关性)
        age_str = patient.get("age", "0")
        gender_str = patient.get("gender", "")

        # 构造特征关键词：如 "4岁 儿童", "65岁 老年人", "孕妇"
        patient_tags = [f"{age_str}"]
        try:
            # 简单的规则提取
            age_val = float(re.search(r"\d+", str(age_str)).group())
            if age_val < 18:
                patient_tags.extend(["儿童", "未成年人", "小儿"])
            if age_val < 1:
                patient_tags.append("婴幼儿")
            if age_val >= 60:
                patient_tags.append("老年人")
        except:
            pass

        if "孕" in str(case_json): patient_tags.append("孕妇")
        if "哺" in str(case_json): patient_tags.append("哺乳期")

        patient_feature_str = " ".join(patient_tags)

        # 3. 构造 Prompt
        # 策略：
        # - 对每个药：生成一条包含【说明书、用法、禁忌、适应症、特殊人群】的综合查询
        # - 对整体：生成一条【药物相互作用、配伍禁忌】的查询 (如果药>1)

        drugs_json_str = json.dumps(drug_names, ensure_ascii=False)

        system_instruction = (
            "你是一个【医学检索查询生成器】。\n"
            "请根据患者情况和药物列表，生成用于 RAG 检索的查询语句列表。\n"
            "必须输出纯 JSON 字符串列表 List[str]，严禁包含 Markdown 标记或解释文字。"
        )

        user_prompt = f"""
        【输入上下文】
        患者特征: {patient_feature_str} {gender_str}
        临床诊断: {diagnosis_str}
        药物列表: {drugs_json_str}

        【生成任务】
        请生成两类查询语句：

        1. **单药综合查询**：请遍历药物列表，对**每一个药物**生成 1 条查询，涵盖该药的说明书核心要素。
           - 格式："[药物名] 说明书 适应症 禁忌症 用法用量 注意事项 {patient_feature_str} 是否适用于 {diagnosis_str}"

        2. **联合用药查询**：如果药物列表中有 2 个或以上药物，请生成 1 条查询用于检查相互作用。
           - 格式："{' '.join(drug_names)} 药物相互作用 配伍禁忌 重复用药风险"
           - (如果只有 1 个药，则不生成此条)

        【示例】
        输入: 药物=["A药", "B药"], 患者="65岁 老年", 诊断="高血压"
        输出: [
            "A药 说明书 适应症 禁忌症 用法用量 注意事项 65岁 老年 是否适用于 高血压",
            "B药 说明书 适应症 禁忌症 用法用量 注意事项 65岁 老年 是否适用于 高血压",
            "A药 B药 药物相互作用 配伍禁忌"
        ]

        【请输出结果 (JSON List Only)】
        """
        try:
            logger.debug(f"正在生成原子查询 (使用 {self.fast_model})...")
            # 使用 fast_model (1.5B) 即可，这主要是格式化任务
            response = self.models.ollama.generate(
                [{"role": "system", "content": system_instruction},
                 {"role": "user", "content": user_prompt}],
                model=self.fast_model
            )

            # --- 4. 增强解析逻辑 (Copied from previous robust implementation) ---
            cleaned_response = response.strip()
            # 去除 Markdown ```json ... ```
            if cleaned_response.startswith("```"):
                lines = cleaned_response.split('\n')
                if len(lines) >= 2:
                    cleaned_response = "\n".join(lines[1:-1])
            cleaned_response = cleaned_response.strip()

            queries = []

            # 策略 A: ast.literal_eval (处理单引号)
            try:
                queries = ast.literal_eval(cleaned_response)
            except:
                pass

            # 策略 B: json.loads + 正则提取
            if not queries:
                match = re.search(r"\[[\s\S]*\]", response)
                if match:
                    try:
                        queries = json.loads(match.group(0))
                    except:
                        try:
                            queries = ast.literal_eval(match.group(0))
                        except:
                            pass

            # 验证与返回
            if queries and isinstance(queries, list) and len(queries) > 0:
                final_queries = [str(q) for q in queries if isinstance(q, str)]
                logger.info(f"成功拆解出 {len(final_queries)} 条查询。")
                return final_queries

            # 失败兜底
            logger.warning(f"原子查询解析失败，原始响应: {response[:100]}...，启用规则降级。")
            return self._fallback_decomposition(meds, patient_feature_str)

        except Exception as e:
            logger.error(f"查询拆解过程发生异常: {e}")
            return self._fallback_decomposition(meds, patient_feature_str)

    def _fallback_decomposition(self, meds: List[dict], patient_str: str = "") -> List[str]:
        """
        [规则降级] 当 LLM 生成失败时，使用固定模板生成查询
        """
        queries = []
        drug_names = []

        # 1. 单药查询
        for m in meds:
            name = m.get('name')
            if name:
                drug_names.append(name)
                # 生成一条大而全的查询，利用 BGE-M3 的长文本检索能力
                q = f"{name} 说明书 适应症 禁忌症 用法用量 注意事项 {patient_str}"
                queries.append(q)

        # 2. 相互作用查询 (如果有多药)
        if len(drug_names) > 1:
            all_drugs = " ".join(drug_names)
            queries.append(f"{all_drugs} 药物相互作用 配伍禁忌")

        return queries

    def _execute_batch_audit(self, queries: List[str], case_context: dict) -> dict:
        """
        [三甲医院标准版] 通用药物审核引擎

        设计哲学：遵循 JCI/HIMSS 药学审核标准，采用"五维审核架构"。
        不再硬编码具体规则，而是赋予 LLM 完整的药学评估逻辑框架。
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []

        # 1. 深度结构化上下文 (Deep Context Extraction)
        # 不仅提取基本信息，还提取共病、肝肾功能、联用药物
        patient = case_context.get("patient_info", {})
        age = patient.get("age", "未知")
        gender = patient.get("gender", "未知")

        # 提取关键体征/检验值 (eGFR 是老年人用药的生命线)
        history = case_context.get("medical_history", {})
        pmh = history.get("past_medical_history") or ""
        hpi = history.get("history_present_illness") or ""
        full_history = f"{pmh}; {hpi}"

        # 提取诊断
        diagnosis_list = case_context.get("diagnosis_info", {}).get("clinical_diagnosis", [])
        diagnosis_str = ", ".join(diagnosis_list) if diagnosis_list else "未明确诊断"

        # 提取处方全貌 (用于相互作用检查)
        med_list = case_context.get("treatment_plan", {}).get("medications", [])
        all_med_names = [m.get('name') for m in med_list if m.get('name')]
        all_meds_str = ", ".join(all_med_names)

        # 2. Worker Function
        def _process_single_query(query: str):
            try:
                # --- 步骤 A: 检索 (RAG) ---
                docs = self._hybrid_retrieve(query)
                if not docs:
                    return {"query": query, "evidence_sources": [], "ai_review": "⚠️ 资料缺失：知识库未收录相关说明书。"}

                # 构造证据文本
                context_parts = []
                for i, d in enumerate(docs):
                    src = d.metadata.get("source", "未知")
                    # 截取更长的片段以包含剂量表
                    content = d.page_content[:600].replace("\n", " ")
                    context_parts.append(f"证据{i + 1} [来源:{src}]: {content}")
                context_text = "\n".join(context_parts)
                sources = list(set([d.metadata.get("source", "未知") for d in docs]))

                # --- 步骤 B: 锁定当前审核目标 ---
                # 从查询中通过简单的切分获取当前正在查的主药
                # 假设查询格式为 "[药名] ..."
                target_drug_name = query.split(" ")[0]

                # 查找该药物在处方中的具体用法 (如 "40mg po qn")
                current_usage = "用法未明"
                for m in med_list:
                    if m.get('name') in target_drug_name or target_drug_name in m.get('name'):
                        spec = m.get('specification', '')
                        usage = m.get('usage', '')
                        current_usage = f"{spec} {usage}".strip()
                        break

                # --- 步骤 C: 构造通用药学 Prompt (核心) ---
                audit_prompt = f"""
                你是一名前置审方系统的**临床药师**。请依据【医学证据】对【当前处方】进行合规性审核。

                【患者档案】
                - 概况：{age} {gender}
                - 诊断：{diagnosis_str}
                - 病史/指标：{full_history} (注意检查肝肾功能、过敏史)

                【当前处方】
                - **正在审核药物**：{target_drug_name}
                - **该药具体用法**：{current_usage}
                - **联合用药环境**：{all_meds_str} (注意检查与这些药的相互作用)

                【医学证据片段】
                {context_text}

                【审核任务：五维评估法】
                请严格基于证据，按以下维度逐一排查。如果证据中未提及某维度，则默认通过。

                1. **适应症匹配 (Indication)**：
                   - 药物是否用于治疗上述【诊断】？不对症即为不合理。

                2. **禁忌症红线 (Contraindication)**：
                   - 检查患者的【年龄】、【性别】、【病史】是否命中说明书中的“禁用”条款。
                   - **无罪推定原则**：如果病历未提及某具体疾病（如胃溃疡），默认患者无此病。

                3. **用法用量审核 (Dosage - 关键)**：
                   - **提取限制**：从证据中寻找“最大剂量”、“老年人剂量”、“肝肾功能不全剂量调整”的描述。
                   - **数值比对**：将【该药具体用法】与证据中的限制进行比对。
                   - **逻辑判定**：例如证据说“与胺碘酮合用不超过20mg”，而处方是“40mg”，则判定为高风险。

                4. **药物相互作用 (Interactions)**：
                   - 检查{target_drug_name}是否与【联合用药环境】中的其他药物有严重冲突（如增加毒性、横纹肌溶解、QT延长）。

                5. **特殊人群 (Special Pop)**：
                   - 针对{age}患者，是否有特殊的“慎用”或“减量”要求？

                【输出规范】
                请输出一段简练的药师审核意见。
                - 必须明确给出风险等级：**高风险** (违反禁忌/超限/致死交互)、**中风险** (慎用/未减量)、**低风险** (正常)。
                - 必须引用证据中的具体数值或条款作为理由。
                """

                # --- 步骤 D: 推理 (Smart Model) ---
                review_res = self.models.ollama.generate(
                    [
                        {"role": "system", "content": "你是一个严谨、客观的药学审核引擎。只根据提供的证据说话。"},
                        {"role": "user", "content": audit_prompt}
                    ],
                    model=self.smart_model  # 必须用 7B+ 模型处理复杂逻辑
                )

                return {"query": query, "evidence_sources": sources, "ai_review": review_res}

            except Exception as e:
                logger.error(f"审核查询 '{query}' 异常: {e}")
                return {"query": query, "ai_review": f"系统错误: {e}", "evidence_sources": []}

        # 3. 并发执行
        logger.info(f"开始并发审核 {len(queries)} 个查询...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_query = {executor.submit(_process_single_query, q): q for q in queries}

            for future in as_completed(future_to_query):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as exc:
                    logger.error(f"任务执行失败: {exc}")

        # 4. 最终决策汇总 (Reduce)
        if not results:
            return {"details": [], "overall_analysis": {"final_decision": "通过", "summary_text": "无风险"}}

        try:
            audit_trace = "\n".join([f"【针对 {r['query'].split(' ')[0]} 的审核】: {r['ai_review']}" for r in results])

            summary_prompt = f"""
            我是科室主任。请根据下属药师的单项报告，汇总一份最终的【处方审核结论】。

            【患者】{age} {gender}，诊断：{diagnosis_str}
            【单项报告】
            {audit_trace}

            【决策逻辑】
            1. 只要有一个“高风险”，最终决策即为**拦截**。
            2. 只有“中风险”且无高风险，决策为**人工复核**。
            3. 全部低风险，决策为**通过**。

            【输出格式 (JSON Only)】
            {{
                "final_decision": "通过 / 拦截 / 人工复核",
                "max_risk_level": "高 / 中 / 低",
                "summary_text": "简明扼要的一句话总结（例如：辛伐他汀剂量超标，存在横纹肌溶解风险，建议拦截）。",
                "actionable_advice": "给医生的具体修改建议。"
            }}
            """

            final_verdict_raw = self.models.ollama.generate(
                [{"role": "system", "content": "输出纯 JSON。"}, {"role": "user", "content": summary_prompt}],
                model=self.smart_model
            )

            import json, re
            match = re.search(r"\{[\s\S]*\}", final_verdict_raw)
            if match:
                overall_analysis = json.loads(match.group(0))
            else:
                overall_analysis = {"final_decision": "人工复核", "summary_text": final_verdict_raw}

        except Exception as e:
            overall_analysis = {"final_decision": "系统错误", "summary_text": str(e)}

        return {
            "details": results,
            "overall_analysis": overall_analysis
        }

    def review_record(self, medical_text: str) -> dict:
        """病历检阅主入口 (Extraction -> Decomposition -> Batch Audit)"""
        logger.info("开始病历检阅（review_record）")

        # 1. 结构化抽取
        prompt = self._build_review_prompt(medical_text)#结构化抽取完全没有必要，在生产环节会自动上游接入
        logger.debug("病历结构化抽取（JSON 已结构化）")

        raw_output = self._safe_llm_call(prompt)#调用ollama client 传入对应的模型
        logger.debug("病历结构化、json化完成")

        extracted_data = self._parse_or_repair_json(raw_output, medical_text)#解析json，生产环境从下面开始

        logger.info("病历检阅完成（JSON 已结构化）%s", extracted_data)

        # 1. 尝试从标准位置获取
        meds = extracted_data.get("treatment_plan", {}).get("medications")#todo his系统中的病历格式

        # 2. 如果没找到，尝试从根目录获取（兼容小模型）
        if not meds:
            meds = extracted_data.get("medications")
            # 如果在根目录找到了，手动重建标准结构，方便后续代码调用
            if meds:
                extracted_data["treatment_plan"] = {"medications": meds}
                logger.info("检测到扁平化 JSON 结构，已自动归一化。")

        # 3. 再次检查
        if not meds:
            logger.warning("未检测到处方信息，跳过用药审核步骤。")
            extracted_data["audit_report"] = "无处方信息，无法审核。"
            return extracted_data

        # 2. 原子化查询拆解
        logger.info("正在进行原子化查询拆解...")
        atomic_queries = self._decompose_case_to_atomic_queries(extracted_data)
        logger.debug(f"生成的原子查询列表: {atomic_queries}")

        # 3. 批量审核 (RAG 辅助)
        #多进程审核
        full_audit_result = self._execute_batch_audit(atomic_queries, extracted_data)

        # 4. 结果注入
        extracted_data["audit_report_details"] = full_audit_result["details"]
        extracted_data["audit_report_summary"] = full_audit_result["overall_analysis"]
        extracted_data["audit_report"] = full_audit_result

        logger.info("病历检阅流程结束。")
        return extracted_data

    # [新增] 暴露给前端的动态配置接口
    def update_config(self, k: int, threshold: float, kn: int, ):
        self.retrieval_k = k
        self.retrieval_threshold = threshold
        self.rerank_top_n = kn
        logger.info(f"配置已更新: K={k}, Threshold={threshold}, rerank_top_n = ={kn}")

    # [新增] 暴露给前端的知识库新增接口
    # 建议使用 add_knowledge_file 替代此方法，或者在这里也加入图谱逻辑
    def add_knowledge(self, text: str, filename: str):
        # 复用 add_knowledge_file 的逻辑
        if self.graph_manager:
            self.graph_manager.add_document(text, filename)
        return self.knowledge.add_document(text, filename)