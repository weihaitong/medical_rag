import logging
import json
import re
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
        # [新增] 动态配置参数 (默认值)
        self.retrieval_k = 10
        self.retrieval_threshold = 0.65
        self.enable_rerank = True

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
        """调用 Ollama Client 生成"""
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

    def _parse_or_repair_json(self, text: str, medical_text: str) -> dict:
        try:
            return json.loads(text)
        except Exception:
            logger.warning("JSON 直接解析失败，尝试修复")

        extracted = self._extract_json_block(text)
        if extracted:
            try:
                return json.loads(extracted)
            except:
                pass

        repaired = self._repair_json_with_llm(text, medical_text)
        try:
            return json.loads(repaired)
        except:
            return self._fallback_empty_review()

    def _decompose_case_to_atomic_queries(self, case_json: dict) -> List[str]:
        """[完全复原] 将病例拆解为原子查询"""
        patient = case_json.get("patient_info", {}) or {}
        diagnosis_info = case_json.get("diagnosis_info", {}) or {}
        clinical_diagnosis = diagnosis_info.get("clinical_diagnosis", []) or []
        treatment_plan = case_json.get("treatment_plan") or {}
        meds = treatment_plan.get("medications") or []

        real_drug_names = []
        valid_meds_str_list = []
        for m in meds:
            if isinstance(m, dict):
                name = m.get('name')
                usage = m.get('usage')
                if name:
                    real_drug_names.append(name)
                    med_str = f"{name} {usage or ''}".strip()
                    valid_meds_str_list.append(med_str)

        meds_context = ", ".join(valid_meds_str_list) if valid_meds_str_list else "无明确处方记录"
        drug_names_str = ", ".join(real_drug_names) if real_drug_names else "无"

        context_str = (
            f"患者: {patient.get('age', '未知')}, {patient.get('gender', '未知')}\n"
            f"诊断: {', '.join(clinical_diagnosis)}\n"
            f"处方: {meds_context}"
        )

        system_instruction = (
            "你是一个专业的【医学检索查询生成器】。\n"
            "你的任务是生成搜索语句（String），而不是提取数据。\n"
            "【严禁】输出 Key-Value 字典或对象。\n"
            "【必须】输出纯字符串列表，例如：[\"查询语句1\", \"查询语句2\"]。"
        )

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
            messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": user_prompt}]
            # 直接调用 ollama，绕过 invoke_llm，和原代码一致
            response = self.models.ollama.generate(messages)

            match = re.search(r"\[[\s\S]*\]", response)
            if match:
                queries = json.loads(match.group(0))
                return [str(q) for q in queries if isinstance(q, str)]
            else:
                return self._fallback_decomposition(meds)
        except Exception as e:
            logger.error(f"查询拆解失败: {e}")
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

    def _execute_batch_audit(self, queries: List[str], case_context: dict) -> dict:
        """
        [完全复原] 执行批量审核与整体总结 (Map-Reduce 逻辑)
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

                # 3. 调用 LLM
                review_res = self.models.ollama.generate([
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
                "overall_analysis": {
                    "final_decision": "无需审核",
                    "max_risk_level": "无",
                    "summary_text": "未生成有效查询或未触发审核规则，系统判断无风险。",
                    "actionable_advice": "无"
                }
            }

        logger.info("单点审核完成，正在生成整体综述报告...")
        try:
            # 1. 准备汇总上下文
            patient_info = case_context.get("patient_info", {})
            diagnosis_str = ", ".join(case_context.get("diagnosis_info", {}).get("clinical_diagnosis", []))

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

            # 3. 调用 LLM
            final_verdict_raw = self.models.ollama.generate([
                {"role": "system", "content": "你是由系统生成的最终决策层，必须输出 JSON 格式。"},
                {"role": "user", "content": summary_prompt}
            ])

            # 4. 解析
            import json, re
            try:
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

    def review_record(self, medical_text: str) -> dict:
        """病历检阅主入口 (Extraction -> Decomposition -> Batch Audit)"""
        logger.info("开始病历检阅（review_record）")

        # 1. 结构化抽取
        prompt = self._build_review_prompt(medical_text)
        raw_output = self._safe_llm_call(prompt)
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
    def update_config(self, k: int, threshold: float):
        self.retrieval_k = k
        self.retrieval_threshold = threshold
        logger.info(f"配置已更新: K={k}, Threshold={threshold}")

    # [新增] 暴露给前端的知识库新增接口
    def add_knowledge(self, text: str, filename: str):
        return self.knowledge.add_document(text, filename)