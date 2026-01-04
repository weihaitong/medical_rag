import streamlit as st
import time
import json
import os
from rag_engine import MedicalRAG
import logging
# ================= 页面配置 =================
st.set_page_config(
    page_title="AI 药师工作台 (图谱增强版)",
    page_icon="🏥",
    layout="wide"
)


# 使用 Streamlit 的缓存机制，确保日志只配置一次，不会因为页面刷新而重复添加
@st.cache_resource
def setup_logging(log_file="logs/medical_rag.log"):
    # 1. 获取根记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 2. 清除已有的 FileHandler (防止重复写入)
    # 注意：不要清除 StreamHandler，否则控制台看不到了
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)

    # 3. 创建新的 FileHandler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 4. 添加到根记录器
    logger.addHandler(file_handler)

    print(f"日志系统已初始化，输出文件: {os.path.abspath(log_file)}")
    return logger


# 执行初始化
setup_logging()

# ================= CSS 美化 =================
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6 }
    .main-header { font-size: 24px; font-weight: bold; color: #2c3e50; }
    .param-box { border: 1px solid #ddd; padding: 10px; border-radius: 5px; background: #eef; }
    /* 调整侧边栏间距 */
    [data-testid="stSidebar"] { padding-top: 2rem; }
    .stButton>button { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ================= 1. 核心引擎加载 (单例模式) =================
@st.cache_resource
def load_engine():
    """初始化 RAG 引擎，全局只执行一次"""
    print("正在初始化 RAG 引擎...")
    # 确保 data 目录存在，用于临时保存上传文件
    if not os.path.exists("data"):
        os.makedirs("data")
    rag = MedicalRAG()
    return rag


# 加载引擎
with st.spinner("正在启动医疗核心引擎 (Graph + Vector)..."):
    engine = load_engine()

# ================= 2. 侧边栏：控制台 =================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/doctor-male--v1.png", width=60)
    st.title("控制台")

    # ------------------------------------------------
    # 模块 A: 病历场景选择
    # ------------------------------------------------
    st.markdown("### 📂 场景选择")

    cases = []
    try:
        if os.path.exists("demo_cases.json"):
            with open("demo_cases.json", "r", encoding="utf-8") as f:
                cases = json.load(f)
        else:
            st.warning("未找到 demo_cases.json")
    except Exception as e:
        st.error(f"加载病历库失败: {e}")

    case_names = [c["title"] for c in cases]
    options = ["-- 自定义输入 --"] + case_names

    selected_case_name = st.selectbox(
        "选择演示病历:",
        options,
        index=0
    )

    default_text = ""
    case_desc = "手动输入或粘贴文本"

    if selected_case_name != "-- 自定义输入 --":
        selected_data = next((c for c in cases if c["title"] == selected_case_name), None)
        if selected_data:
            default_text = selected_data.get("content", "")
            case_desc = selected_data.get("description", "")

    st.caption(f"当前场景：{case_desc}")
    st.divider()

    # ------------------------------------------------
    # 模块 B: 参数调优
    # ------------------------------------------------
    st.markdown("### 🎛️ 模型参数调优")
    with st.container():
        new_threshold = st.slider(
            "相似度阈值 (Threshold)",
            min_value=0.0, max_value=1.0,
            value=engine.retrieval_threshold,
            step=0.05
        )

        new_k = st.slider(
            "初筛数量 (Retrieval K)",
            min_value=5, max_value=50,
            value=engine.retrieval_k,
            step=5
        )

        new_rerank_n = st.slider(
            "重排数量 (Rerank Top-N)",
            min_value=1, max_value=15,
            value=getattr(engine, 'rerank_top_n', 5),
            step=1
        )

        if (new_threshold != engine.retrieval_threshold or
                new_k != engine.retrieval_k or
                new_rerank_n != engine.rerank_top_n):
            engine.update_config(k=new_k, threshold=new_threshold, kn=new_rerank_n)
            st.toast(f"参数已更新", icon="✅")

    st.divider()

    # ------------------------------------------------
    # 模块 C: 知识库管理 (新增/删除) - 核心修改部分
    # ------------------------------------------------
    st.markdown("### 📚 知识库管理")

    # 使用 Tabs 区分新增和删除操作
    kb_tab1, kb_tab2 = st.tabs(["📥 新增说明书", "🗑️ 删除药品"])

    # === Tab 1: 新增说明书 ===
    with kb_tab1:
        st.caption("支持 TXT/MD，将自动同步至图谱与向量库")

        # 方式 1: 文件上传
        uploaded_file = st.file_uploader("上传文件", type=["txt", "md"], label_visibility="collapsed")

        # 方式 2: 文本粘贴
        manual_text = st.text_area("或直接粘贴内容", height=100, placeholder="粘贴药品说明书全文...")
        manual_name = st.text_input("文档标题 (例如: 阿莫西林说明书)", placeholder="必填，带后缀如 .txt")

        if st.button("提交入库", key="btn_add", use_container_width=True):
            target_path = ""

            # 1. 确定文件保存路径
            if uploaded_file:
                target_path = os.path.join("data", uploaded_file.name)
                with open(target_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            elif manual_text and manual_name:
                # 自动补全 .txt
                if not manual_name.endswith((".txt", ".md")):
                    manual_name += ".txt"
                target_path = os.path.join("data", manual_name)
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(manual_text)

            # 2. 调用后端接口
            if target_path:
                try:
                    with st.spinner("正在解析文本并构建图谱..."):
                        # 调用 MedicalRAG 的 add_knowledge_file
                        result = engine.add_knowledge_file(target_path)

                    # 结果展示
                    if result.get("graph_db") == "Success":
                        st.success(f"✅ 入库成功！\nVector: {result['vector_db']} | Graph: {result['graph_db']}")
                        time.sleep(1)
                    else:
                        st.error(f"❌ 入库失败: {result}")
                except Exception as e:
                    st.error(f"处理异常: {e}")
            else:
                st.warning("请上传文件或填写完整内容与标题")

    # === Tab 2: 删除药品 ===
    with kb_tab2:
        st.caption("从图数据库中移除药品实体及其关系")
        drug_to_del = st.text_input("药品通用名", placeholder="例如：左氧氟沙星片")

        if st.button("执行删除", key="btn_del", type="secondary", use_container_width=True):
            if drug_to_del:
                with st.spinner(f"正在移除 {drug_to_del}..."):
                    try:
                        # 调用 MedicalRAG 的 delete_drug_knowledge
                        res_msg = engine.delete_drug_knowledge(drug_to_del)
                        st.info(res_msg)
                    except Exception as e:
                        st.error(f"删除失败: {e}")
            else:
                st.warning("请输入药品名称")

# ================= 3. 主界面：病历审核 =================

st.markdown('<div class="main-header">🏥 智能处方审核系统 (Graph-RAG)</div>', unsafe_allow_html=True)

st.info(
    f"⚙️ 引擎状态：Graph DB [Online] | Vector DB [Online] | 配置: Thr={engine.retrieval_threshold}, TopN={engine.rerank_top_n}")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 病历输入")

    medical_input = st.text_area(
        "病历文本 (支持编辑)",
        value=default_text,
        height=400,
        help="在此输入或编辑病历信息"
    )

    audit_btn = st.button("🚀 开始智能审核", type="primary", use_container_width=True)

if audit_btn:
    if not medical_input.strip():
        st.warning("请输入病历文本或在左侧选择演示案例。")
    else:
        with col2:
            st.subheader("🔍 审核报告")

            status_box = st.status("正在进行全链路审查...", expanded=True)

            try:
                # 1. 结构化
                status_box.write("1. 正在整理病历与查询生成...")

                # 2. 检索
                status_box.write(f"2. 双路召回 (图谱推理 + 向量相似度)...")

                # 3. 审核
                status_box.write("3. 正在生成药学决策...")

                start_time = time.time()
                result = engine.review_record(medical_input)
                end_time = time.time()

                status_box.update(label=f"审核完成 (耗时 {end_time - start_time:.2f}s)", state="complete",
                                  expanded=False)

                # === 展示结果 ===
                summary = result.get("audit_report_summary", {})
                decision = summary.get("final_decision", "未知")

                # 颜色逻辑
                color = "#28a745"  # Green
                if "拦截" in decision or "禁用" in decision or "高风险" in decision:
                    color = "#dc3545"  # Red
                elif "人工" in decision or "慎用" in decision or "中风险" in decision:
                    color = "#ffc107"  # Orange

                st.markdown(f"""
                <div style="padding: 15px; border-left: 5px solid {color}; background-color: #f9f9f9; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h3 style="color: {color}; margin:0;">🛡️ {decision}</h3>
                    <p style="margin-top:10px; font-weight:bold;">综合评价：</p>
                    <p>{summary.get("summary_text", "无")}</p>
                    <p style="margin-top:10px; font-weight:bold;">建议操作：</p>
                    <p>{summary.get("actionable_advice", "无")}</p>
                </div>
                """, unsafe_allow_html=True)

                st.divider()

                # 详情与证据
                st.markdown("#### 🔬 风险详情与证据链")
                details = result.get("audit_report_details", [])

                if not details:
                    st.info("未发现具体的风险点或未触发检索。")

                for item in details:
                    ai_review = item.get('ai_review', '')

                    # 简单的状态图标判定
                    icon = "✅"
                    risk_class = "low-risk"
                    if any(x in ai_review for x in ["高风险", "禁忌", "拦截", "禁用"]):
                        icon = "🔴"
                        risk_class = "high-risk"
                    elif any(x in ai_review for x in ["中风险", "慎用", "未知"]):
                        icon = "🟠"
                        risk_class = "medium-risk"

                    with st.expander(f"{icon} 检查项：{item['query']}", expanded=True):
                        st.markdown(f"**AI 结论：**\n\n{ai_review}")

                        st.markdown("---")
                        st.caption("📚 证据来源：")
                        sources = item.get('evidence_sources', [])

                        # 区分图谱来源和向量库来源
                        if sources:
                            for s in sources:
                                if s == "neo4j":
                                    st.markdown("🔹 `知识图谱 (Neo4j)` :blue[结构化关系推理]")
                                elif s == "❌ 知识库缺失":
                                    st.markdown("❌ `无相关资料`")
                                else:
                                    st.markdown(f"📄 `{s}` :grey[说明书文档]")
                        else:
                            st.text("无相关来源")

                with st.expander("🛠️ 查看原始 JSON 数据"):
                    st.json(result)

            except Exception as e:
                st.error(f"系统运行错误: {e}")
                status_box.update(label="审核失败", state="error")
                st.exception(e)

#streamlit run demo_app.py 运行demo_app
