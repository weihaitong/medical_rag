import logging
import json
import re
import ast
from typing import List, Optional

from langchain_core.documents import Document
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 引入解耦的模块
from model_factory import ModelFactory
from knowledge_manager import KnowledgeManager

# 图检索支持
try:
    from graph_retriever import GraphRetriever

    GRAPH_RETRIEVER_AVAILABLE = True
except ImportError:
    GRAPH_RETRIEVER_AVAILABLE = False

logger = logging.getLogger(__name__)


class MedicalRAG:
    """RAG 引擎：保留核心业务逻辑，协调 ModelFactory 和 KnowledgeManager"""

    def __init__(self, data_path: str = "data/", collection_name: str = "medical_db"):
        # 1. 引用模型工厂
        self.models = ModelFactory()
        # 2. 引用知识库管理
        self.knowledge = KnowledgeManager(
            data_path=data_path,
            embedding_model=self.models.embedding_model,
            collection_name=collection_name
        )
        # 3. 初始化图检索
        self.graph_retriever = GraphRetriever() if GRAPH_RETRIEVER_AVAILABLE else None
        # 4. 构建业务链
        self._build_rag_chain()
        # 动态配置参数 (默认值)
        self.retrieval_k = 20
        self.retrieval_threshold = 0.65
        self.enable_rerank = True
        self.rerank_top_n = 10

    # =========================================================================
    #  业务功能 1: 混合检索 (完全复原)
    # =========================================================================
    def _hybrid_retrieve(self, query: str) -> List[Document]:
        """执行双路检索与统一重排 (Hybrid Retrieval & Rerank)"""
        logger.info(f"--- 开始双路检索: {query} ---")
        all_candidates: List[Document] = []

        # Path A: 图引擎
        if self.graph_retriever:
            try:
                graph_docs = self.graph_retriever.retrieve(query)
                for doc in graph_docs:
                    doc.metadata["source_type"] = "KnowledgeGraph"
                    doc.metadata["confidence"] = "high"
                all_candidates.extend(graph_docs)
                logger.info(f"Path A (Graph) 命中 {len(graph_docs)} 条结构化证据")
            except Exception as e:
                logger.warning(f"图引擎检索异常: {e}")

        # Path B: 向量引擎
        if self.knowledge.vector_store:
            try:
                results_with_score = self.knowledge.vector_store.similarity_search_with_score(query, k=self.retrieval_k)
                valid_docs = []
                # [修改] 使用动态阈值
                current_threshold = self.retrieval_threshold
                for doc, score in results_with_score:
                    if score >= current_threshold:
                        doc.metadata["source_type"] = "VectorDB"
                        valid_docs.append(doc)
                all_candidates.extend(valid_docs)
                logger.info(f"Path B (Vector) 召回 {len(valid_docs)} 条非结构化片段")
            except Exception as e:
                logger.error(f"向量检索异常: {e}")

        # Path C: 统一重排
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
            compressor = CrossEncoderReranker(model=self.models.reranker, top_n=3)
            reranked_docs = compressor.compress_documents(documents=candidates_list, query=query)
            # 调试日志：看看最终选了啥
            for i, d in enumerate(reranked_docs):
                src = d.metadata.get("source", "未知")
                content_preview = d.page_content[:50].replace("\n", " ")
                logger.debug(f"Top-{i+1} [{src}]: {content_preview}...")

            return reranked_docs
        except Exception as e:
            logger.error(f"Rerank 失败，降级为原始顺序: {e}")
            return candidates_list[:3]

    def _rewrite_query(self, query: str) -> str:
        """基于规则的简单重写"""
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
        """构建问答链"""

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
    #  业务功能 2: 病历检阅 (Review Record) - 完全复原
    # =========================================================================

    def _build_review_prompt(self, medical_text: str) -> str:
        """
        [完全复原] 构建病历检阅 Prompt，包含所有原有字段。
        """
        return f"""
        你是一名资深的病历结构化专家。请阅读下方的【待处理病历文本】，提取关键信息填入指定的 JSON 格式。

        【提取规则】
        1. **完整性**：如果处方中有多种药物，**必须全部提取**，存入 medications 列表，严禁遗漏。
        2. **准确性**：数值和单位必须与原文一致。
        3. **格式**：只输出标准的 JSON 字符串，不要包含 Markdown 标记（如 ```json），不要包含注释。
        4. **空值处理**：未提及的字段填 null。
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
        return self.models.ollama.generate(messages)

    def _safe_llm_call(self, prompt: str, retry: int = 2) -> str:
        last_error = None
        for attempt in range(1, retry + 1):
            try:
                return self._invoke_llm(prompt)
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

    # def _parse_or_repair_json(self, text: str, medical_text: str) -> dict:
    #     """
    #             解析 JSON，具备极强的容错能力：
    #             1. 去除 Markdown
    #             2. 尝试标准 JSON 解析 (json.loads)
    #             3. 尝试 Python 字面量解析 (ast.literal_eval) -> 解决 None/True/False/单引号问题
    #             4. 尝试正则修正 (None->null)
    #             5. LLM 修复
    #             """
    #     # --- 预处理：去除 Markdown 标记 ---
    #     cleaned_text = text.strip()
    #     # 去除 ```json ... ``` 或 ``` ... ```
    #     if cleaned_text.startswith("```"):
    #         # 找到第一个换行符
    #         first_newline = cleaned_text.find("\n")
    #         if first_newline != -1:
    #             cleaned_text = cleaned_text[first_newline + 1:]
    #         if cleaned_text.endswith("```"):
    #             cleaned_text = cleaned_text[:-3]
    #     cleaned_text = cleaned_text.strip()
    #
    #     try:
    #         return json.loads(cleaned_text)
    #     except Exception:
    #         logger.warning("JSON 直接解析失败，尝试修复")
    #         pass
    #     extracted = self._extract_json_block(text)
    #     if extracted:
    #         try:
    #             return json.loads(extracted)
    #         except:
    #             pass
    #
    #     repaired = self._repair_json_with_llm(text, medical_text)
    #     try:
    #         return json.loads(repaired)
    #     except:
    #         return self._fallback_empty_review()
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

    # def _decompose_case_to_atomic_queries(self, case_json: dict) -> List[str]:
    #     """[完全复原] 将病例拆解为原子查询"""
    #     patient = case_json.get("patient_info", {}) or {}
    #     diagnosis_info = case_json.get("diagnosis_info", {}) or {}
    #     clinical_diagnosis = diagnosis_info.get("clinical_diagnosis", []) or []
    #     treatment_plan = case_json.get("treatment_plan") or {}
    #     meds = treatment_plan.get("medications") or []
    #
    #     real_drug_names = []
    #     valid_meds_str_list = []
    #     for m in meds:
    #         if isinstance(m, dict):
    #             name = m.get('name')
    #             usage = m.get('usage')
    #             if name:
    #                 real_drug_names.append(name)
    #                 med_str = f"{name} {usage or ''}".strip()
    #                 valid_meds_str_list.append(med_str)
    #
    #     meds_context = ", ".join(valid_meds_str_list) if valid_meds_str_list else "无明确处方记录"
    #     drug_names_str = ", ".join(real_drug_names) if real_drug_names else "无"
    #
    #     context_str = (
    #         f"患者: {patient.get('age', '未知')}, {patient.get('gender', '未知')}\n"
    #         f"诊断: {', '.join(clinical_diagnosis)}\n"
    #         f"处方: {meds_context}"
    #     )
    #
    #     system_instruction = (
    #         "你是一个专业的【医学检索查询生成器】。\n"
    #         "你的任务是生成搜索语句（String），而不是提取数据。\n"
    #         "【严禁】输出 Key-Value 字典或对象。\n"
    #         "【必须】输出纯字符串列表，例如：[\"查询语句1\", \"查询语句2\"]。"
    #     )
    #
    #     user_prompt = f"""
    #     请将以下病例拆解为用于向量检索的【原子查询语句】。
    #
    #     【病例信息】
    #     {context_str}
    #
    #     【当前药物列表】
    #     {meds_context}
    #
    #     【生成任务】(请严格执行，不要输出对象，只输出句子)
    #     1. 用法用量：
    #        - 生成：{drug_names_str}在{patient.get('age', '该年龄段')}中的用法用量
    #     2. 禁忌症：
    #        - 生成：{drug_names_str}的禁忌症
    #     3. 适应症匹配：
    #        - 对每个诊断生成：{drug_names_str}是否适用于[诊断]
    #     4. 相互作用：
    #        - (如果处方只有1种药，请忽略此项，不要生成null)
    #
    #     【格式强制要求】
    #     - 输出必须是 JSON 字符串列表 (List[str])。
    #     - 列表中的元素必须是完整的自然语言句子。
    #     - 正确格式：["乳果糖的用法用量", "乳果糖是否适用于便秘"]
    #
    #     【请直接输出结果，不要包含Markdown标记】
    #     """
    #
    #     try:
    #         logger.debug("正在生成原子查询...")
    #         messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": user_prompt}]
    #         # 直接调用 ollama，绕过 invoke_llm，和原代码一致
    #         response = self.models.ollama.generate(messages)
    #
    #         match = re.search(r"\[[\s\S]*\]", response)
    #         if match:
    #             queries = json.loads(match.group(0))
    #             return [str(q) for q in queries if isinstance(q, str)]
    #         else:
    #             return self._fallback_decomposition(meds)
    #     except Exception as e:
    #         logger.error(f"查询拆解失败: {e}")
    #         return self._fallback_decomposition(meds)
    # def _decompose_case_to_atomic_queries(self, case_json: dict) -> List[str]:
    #     """
    #     [优化版] 将病例拆解为原子查询
    #     策略：强制要求 LLM 对【每个药物】生成独立查询，禁止合并，提高检索命中率。
    #     """
    #     patient = case_json.get("patient_info", {}) or {}
    #     diagnosis_info = case_json.get("diagnosis_info", {}) or {}
    #     clinical_diagnosis = diagnosis_info.get("clinical_diagnosis", []) or []
    #     treatment_plan = case_json.get("treatment_plan") or {}
    #     meds = treatment_plan.get("medications") or []
    #
    #     # 提取药物名称列表
    #     real_drug_names = []
    #     for m in meds:
    #         if isinstance(m, dict) and m.get('name'):
    #             real_drug_names.append(m.get('name'))
    #
    #     # 如果没有药物，直接返回
    #     if not real_drug_names:
    #         return []
    #
    #     # 构造上下文
    #     patient_str = f"{patient.get('age', '未知年龄')} {patient.get('gender', '未知性别')}"
    #     diagnosis_str = ', '.join(clinical_diagnosis)
    #     drugs_json_str = json.dumps(real_drug_names, ensure_ascii=False)  # ["布洛芬", "左氧氟沙星"]
    #
    #     system_instruction = (
    #         "你是一个医学检索查询生成器。\n"
    #         "你的任务是为列表中的【每一个药物】生成独立的检索语句。\n"
    #         "【严禁】将多个药物合并在同一个查询中（如'A和B的禁忌'是错误的）。\n"
    #         "【必须】输出纯字符串列表 List[str]。"
    #     )
    #
    #     user_prompt = f"""
    #     请根据以下信息生成原子化检索查询。
    #
    #     【患者信息】{patient_str}
    #     【诊断信息】{diagnosis_str}
    #     【药物列表】{drugs_json_str}
    #
    #     【生成规则】
    #     请遍历【药物列表】，对**每一个药物**分别生成以下3个维度的查询语句：
    #     1. 用法用量："[药物名] 说明书 用法用量 儿童/老人/孕妇" (根据患者特征调整)
    #     2. 禁忌症："[药物名] 禁忌症"
    #     3. 相互作用："[药物名] 与其他药物的相互作用" (如果只有1种药则跳过此项)
    #     4. 适应症："[药物名] 是否适用于 [诊断]"
    #
    #     【示例】
    #     输入药物: ["阿莫西林", "布洛芬"]，患者: 3岁
    #     输出: [
    #         "阿莫西林在3岁儿童中的用法用量",
    #         "阿莫西林禁忌症",
    #         "阿莫西林是否适用于感冒",
    #         "布洛芬在3岁儿童中的用法用量",
    #         "布洛芬禁忌症",
    #         "布洛芬是否适用于感冒",
    #         "阿莫西林与布洛芬的相互作用"
    #     ]
    #
    #     【请直接输出 JSON 列表】
    #     """
    #
    #     try:
    #         logger.debug("正在生成原子查询 (One-by-One Strategy)...")
    #         messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": user_prompt}]
    #         response = self.models.ollama.generate(messages)
    #
    #         match = re.search(r"\[[\s\S]*\]", response)
    #         if match:
    #             queries = json.loads(match.group(0))
    #             # 再次清洗，确保都是字符串
    #             return [str(q) for q in queries if isinstance(q, str)]
    #         else:
    #             return self._fallback_decomposition(meds)
    #     except Exception as e:
    #         logger.error(f"查询拆解失败: {e}")
    #         return self._fallback_decomposition(meds)

    def _decompose_case_to_atomic_queries(self, case_json: dict) -> List[str]:
        """
        [增强版] 将病例拆解为原子查询
        增强了解析逻辑，防止因格式问题导致返回空列表。
        """
        import ast  # 确保引用

        patient = case_json.get("patient_info", {}) or {}
        diagnosis_info = case_json.get("diagnosis_info", {}) or {}
        clinical_diagnosis = diagnosis_info.get("clinical_diagnosis", []) or []
        treatment_plan = case_json.get("treatment_plan") or {}
        meds = treatment_plan.get("medications") or []

        # 提取药物名
        drug_names = []
        for m in meds:
            if isinstance(m, dict) and m.get('name'):
                drug_names.append(m.get('name'))

        if not drug_names:
            logger.warning("未提取到药物名称，跳过拆解。")
            return []

        # 构造 Context
        context_str = (
            f"患者: {patient.get('age', '未知')} {patient.get('gender', '未知')}\n"
            f"诊断: {', '.join(clinical_diagnosis)}\n"
            f"药物列表: {', '.join(drug_names)}"
        )

        system_instruction = (
            "你是一个【医学检索查询生成器】。请为列表中的**每一个药物**分别生成检索查询。\n"
            "必须输出纯 JSON 字符串列表 List[str]，不要包含任何解释性文字。"
        )

        # 1. 提取患者特征并进行泛化
        age_str = patient.get("age", "0")
        age_keywords = [f"{age_str}岁"]

        # 简单的规则泛化
        try:
            age_val = float(re.search(r"\d+", str(age_str)).group())
            if age_val < 18:
                age_keywords.extend(["儿童", "未成年人", "18岁以下", "少年"])
            if age_val < 12:
                age_keywords.append("小儿")
            if age_val < 1:
                age_keywords.append("婴幼儿")
            if age_val > 60:
                age_keywords.append("老年人")
        except:
            pass

        age_query_suffix = " ".join(age_keywords)  # 结果如: "4岁 儿童 未成年人 18岁以下"
        user_prompt = f"""
        请将以下病例拆解为原子查询。

        【病例上下文】
        {context_str}

        【生成要求】
        请遍历药物列表 {drug_names}，对**每一个药物**都生成以下 3 条查询：
        1. [药物名] 说明书 用法用量 {age_query_suffix} (已注入年龄泛化词)
        2. [药物名] 禁忌症 {age_query_suffix}
        3. [药物名] 是否适用于 [诊断]
        
        【示例】
        输入: ["A药", "B药"]
        输出: [
            "A药 说明书 用法用量", "A药 禁忌症", "A药 是否适用于感冒",
            "B药 说明书 用法用量", "B药 禁忌症", "B药 是否适用于感冒"
        ]

        【请输出结果 (JSON List Only)】
        """

        try:
            logger.debug("正在生成原子查询...")
            messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": user_prompt}]
            response = self.models.ollama.generate(messages)

            # --- 增强解析逻辑 ---
            # 1. 尝试清洗 Markdown
            cleaned_response = response.strip()
            if cleaned_response.startswith("```"):
                # 去除第一行和最后一行
                lines = cleaned_response.split('\n')
                if len(lines) >= 2:
                    cleaned_response = "\n".join(lines[1:-1])
            cleaned_response = cleaned_response.strip()

            queries = []

            # 2. 尝试 ast.literal_eval (最强解析，支持单引号)
            try:
                queries = ast.literal_eval(cleaned_response)
            except:
                pass

            # 3. 如果失败，尝试正则提取 [...]
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

            # 4. 验证结果有效性
            if queries and isinstance(queries, list) and len(queries) > 0:
                logger.info(f"成功拆解出 {len(queries)} 条查询。")
                return [str(q) for q in queries if isinstance(q, str)]

            # 5. 如果上面都失败了，走降级策略
            logger.warning(f"原子查询解析失败，原始响应: {response[:100]}...，启用规则降级。")
            return self._fallback_decomposition(meds)

        except Exception as e:
            logger.error(f"查询拆解过程发生异常: {e}")
            return self._fallback_decomposition(meds)

    def _fallback_decomposition(self, meds: List[dict]) -> List[str]:
        queries = []
        for m in meds:
            name = m.get('name')
            if name:
                queries.append(f"{name} 说明书 用法用量")
                queries.append(f"{name} 禁忌症")
                queries.append(f"{name} 药物相互作用")
        return queries

    # def _execute_batch_audit(self, queries: List[str], case_context: dict) -> dict:
    #     """
    #     [完全复原] 执行批量审核与整体总结 (Map-Reduce 逻辑)
    #     """
    #     results = []
    #
    #     # --- 阶段 1: 单点审核 (Map) ---
    #     for query in queries:
    #         try:
    #             # 1. 复用混合检索
    #             docs = self._hybrid_retrieve(query)
    #             # 【关键修改】: 召回为空时的兜底逻辑
    #             if not docs:
    #                 logger.warning(f"查询 '{query}' 未召回任何文档，触发安全警告。")
    #                 results.append({
    #                     "query": query,
    #                     "evidence_sources": ["❌ 知识库无相关数据"],
    #                     "ai_review": "⚠️ **系统警告**：当前知识库中未找到该药物/症状的相关说明书或指南，无法进行智能评估。请药师务必**人工核查**此项。"
    #                 })
    #                 continue
    #
    #             # 2. 构造单点审核 Prompt
    #             context_text = "\n".join([d.page_content[:300] for d in docs])
    #             sources = list(set([d.metadata.get("source", "未知") for d in docs]))
    #
    #             audit_prompt = f"""
    #             你是一名药品安全审核员。请依据证据对当前查询进行风险评估。
    #             【当前查询】: {query}
    #             【医学证据】: {context_text}
    #             【任务】: 判断是否存在用药风险。
    #             【输出要求】: 简练的一句话结论，指明风险等级(高/中/低/无)。
    #             """
    #
    #             # 3. 调用 LLM
    #             review_res = self.models.ollama.generate([
    #                 {"role": "system", "content": "你是一个严谨的药学审核助手。请简练回答。"},
    #                 {"role": "user", "content": audit_prompt}
    #             ])
    #
    #             results.append({
    #                 "query": query,
    #                 "evidence_sources": sources,
    #                 "ai_review": review_res
    #             })
    #
    #         except Exception as e:
    #             logger.error(f"审核查询 '{query}' 时发生错误: {e}")
    #
    #     # --- 阶段 2: 整体总结 (Reduce) ---
    #     if not results:
    #         return {
    #             "details": [],
    #             "overall_analysis": {
    #                 "final_decision": "无需审核",
    #                 "max_risk_level": "无",
    #                 "summary_text": "未生成有效查询或未触发审核规则，系统判断无风险。",
    #                 "actionable_advice": "无"
    #             }
    #         }
    #
    #     logger.info("单点审核完成，正在生成整体综述报告...")
    #     try:
    #         # 1. 准备汇总上下文
    #         patient_info = case_context.get("patient_info", {})
    #         diagnosis_str = ", ".join(case_context.get("diagnosis_info", {}).get("clinical_diagnosis", []))
    #
    #         audit_trace = "\n".join([
    #             f"- 检查点: {r['query']}\n  AI发现: {r['ai_review']}"
    #             for r in results
    #         ])
    #
    #         # 2. 构造“主审药师” Prompt
    #         summary_prompt = f"""
    #         你是一名三甲医院的【主任药师】。请根据下方的【患者信息】和【单项审核记录】，生成一份最终的用药安全综合评估报告。
    #
    #         【患者信息】
    #         年龄: {patient_info.get('age', '未知')}, 性别: {patient_info.get('gender', '未知')}
    #         诊断: {diagnosis_str}
    #
    #         【系统单项审核记录】
    #         {audit_trace}
    #
    #         【你的任务】
    #         1. 综合分析所有检查结果，判断是否存在冲突（例如：一个通过，另一个提示高风险）。
    #         2. 给出最终的决策建议（通过 / 拦截 / 提示医生慎用）。
    #         3. 如果有风险，请按照严重程度排序说明。
    #
    #         【输出格式 (JSON)】
    #         请直接输出合法的 JSON，不要包含 Markdown 标记：
    #         {{
    #             "final_decision": "通过/拦截/人工复核",
    #             "max_risk_level": "高/中/低/无",
    #             "summary_text": "简短的综合评价（100字以内）",
    #             "actionable_advice": "给医生的具体建议（如：建议停用XX药，改用XX）"
    #         }}
    #         """
    #
    #         # 3. 调用 LLM
    #         final_verdict_raw = self.models.ollama.generate([
    #             {"role": "system", "content": "你是由系统生成的最终决策层，必须输出 JSON 格式。"},
    #             {"role": "user", "content": summary_prompt}
    #         ])
    #
    #         # 4. 解析
    #         import json, re
    #         try:
    #             match = re.search(r"\{[\s\S]*\}", final_verdict_raw)
    #             if match:
    #                 overall_analysis = json.loads(match.group(0))
    #             else:
    #                 overall_analysis = {"raw_text": final_verdict_raw}
    #         except:
    #             overall_analysis = {"raw_text": final_verdict_raw, "parse_error": "JSON解析失败"}
    #
    #     except Exception as e:
    #         logger.error(f"生成整体总结时出错: {e}")
    #         overall_analysis = {"error": str(e)}
    #
    #     return {
    #         "details": results,
    #         "overall_analysis": overall_analysis
    #     }

    def _execute_batch_audit(self, queries: List[str], case_context: dict) -> dict:
        """
        [防幻觉优化版] 执行批量审核
        核心改进：防止 LLM 张冠李戴（把 A 药的禁忌安在 B 药头上）。
        """
        results = []

        # 提取患者信息
        patient_info = case_context.get("patient_info", {})
        age_str = patient_info.get("age", "未知")
        gender = patient_info.get("gender", "未知")

        for query in queries:
            try:
                # 1. 检索
                docs = self._hybrid_retrieve(query)

                if not docs:
                    results.append({
                        "query": query,
                        "evidence_sources": ["❌ 知识库缺失"],
                        "ai_review": "⚠️ **资料缺失**：未检索到说明书，无法判断。"
                    })
                    continue

                # 2. 构造带来源的上下文 (Key Change!)
                # 格式：[来源: 药品A说明书.txt] ...内容...
                context_parts = []
                for i, d in enumerate(docs):
                    src = d.metadata.get("source", "未知来源")
                    content = d.page_content[:400].replace("\n", " ")  # 压缩换行
                    context_parts.append(f"片段{i + 1} (来源: {src}): {content}")

                context_text = "\n\n".join(context_parts)
                sources = list(set([d.metadata.get("source", "未知") for d in docs]))

                # 从 Query 中提取当前正在查的药物名（简单提取）
                target_drug = query.split(" ")[0]  # 假设 Query 格式为 "药物名 ..."

                # 3. 构造防幻觉 Prompt
                audit_prompt = f"""
                你是一名临床药师。请审核【{target_drug}】在患者（{age_str}, {gender}）身上的用药风险。

                【医学证据片段】
                {context_text}

                【审查步骤】
                1. **来源核对（关键）**：请检查每个片段的“来源”或内容中的“药品名称”。
                   - 如果片段是关于【{target_drug}】的，请采信。
                   - 如果片段是关于其他药物（如左氧氟沙星、阿司匹林等）的，**请直接忽略，严禁引用**。
                2. **提取限制**：仅从核对无误的片段中，寻找关于“年龄”、“禁忌”的描述。
                3. **判定风险**：
                   - 高风险：证据明确说“禁用”。
                   - 中风险：证据说“慎用”或“未进行实验”。
                   - 低风险：用法明确且适用。

                【输出要求】
                简练回答，格式：“风险等级：X。理由：...”。引用证据时请注明片段来源。
                """

                # 4. 调用 LLM
                review_res = self.models.ollama.generate([
                    {"role": "system", "content": "你是一个严谨的药师。注意区分不同药物的说明书，不要张冠李戴。"},
                    {"role": "user", "content": audit_prompt}
                ])

                results.append({
                    "query": query,
                    "evidence_sources": sources,
                    "ai_review": review_res
                })

            except Exception as e:
                logger.error(f"审核查询 '{query}' 时发生错误: {e}")
                results.append({"query": query, "ai_review": f"Error: {e}", "evidence_sources": []})

        # --- 阶段 3: 总结 (保持不变) ---
        if not results:
            return {"details": [], "overall_analysis": {"final_decision": "通过", "summary_text": "无"}}

        try:
            audit_trace = "\n".join([f"- {r['query']} -> {r['ai_review']}" for r in results])
            summary_prompt = f"""
            汇总药师审核结果，给出最终处方建议。
            【患者】{age_str} {gender}
            【审核记录】
            {audit_trace}

            【决策】高风险/禁用 -> 拦截；中风险 -> 人工复核；低风险 -> 通过。

            【输出 JSON】
            {{ "final_decision": "...", "max_risk_level": "...", "summary_text": "...", "actionable_advice": "..." }}
            """

            final_verdict_raw = self.models.ollama.generate([
                {"role": "system", "content": "输出纯 JSON。"},
                {"role": "user", "content": summary_prompt}
            ])

            import json, re, ast
            match = re.search(r"\{[\s\S]*\}", final_verdict_raw)
            if match:
                try:
                    overall_analysis = json.loads(match.group(0))
                except:
                    overall_analysis = ast.literal_eval(match.group(0))
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
        prompt = self._build_review_prompt(medical_text)
        logger.debug("病历结构化抽取（JSON 已结构化）")

        raw_output = self._safe_llm_call(prompt)
        logger.debug("病历结构化、json化完成")

        extracted_data = self._parse_or_repair_json(raw_output, medical_text)

        logger.info("病历检阅完成（JSON 已结构化）%s", extracted_data)

        if not extracted_data.get("treatment_plan", {}).get("medications"):
            logger.warning("未检测到处方信息，跳过用药审核步骤。")
            extracted_data["audit_report"] = "无处方信息，无法审核。"
            return extracted_data

        # 2. 原子化查询拆解
        logger.info("正在进行原子化查询拆解...")
        atomic_queries = self._decompose_case_to_atomic_queries(extracted_data)
        logger.debug(f"生成的原子查询列表: {atomic_queries}")

        # 3. 批量审核 (RAG 辅助)
        full_audit_result = self._execute_batch_audit(atomic_queries, extracted_data)

        # 4. 结果注入
        extracted_data["audit_report_details"] = full_audit_result["details"]
        extracted_data["audit_report_summary"] = full_audit_result["overall_analysis"]
        extracted_data["audit_report"] = full_audit_result

        logger.info("病历检阅流程结束。")
        return extracted_data

# [新增] 暴露给前端的动态配置接口
    def update_config(self, k: int, threshold: float, kn: int,):
        self.retrieval_k = k
        self.retrieval_threshold = threshold
        self.rerank_top_n = kn
        logger.info(f"配置已更新: K={k}, Threshold={threshold}, rerank_top_n = ={kn}")

    # [新增] 暴露给前端的知识库新增接口
    def add_knowledge(self, text: str, filename: str):
        return self.knowledge.add_document(text, filename)