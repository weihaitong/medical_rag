import streamlit as st
import time
import json
import os
from rag_engine import MedicalRAG

# ================= 页面配置 =================
st.set_page_config(
    page_title="AI 药师工作台 (动态配置版)",
    page_icon="🏥",
    layout="wide"
)

# ================= CSS 美化 =================
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6 }
    .main-header { font-size: 24px; font-weight: bold; color: #2c3e50; }
    .param-box { border: 1px solid #ddd; padding: 10px; border-radius: 5px; background: #eef; }
    /* 调整侧边栏间距 */
    [data-testid="stSidebar"] { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ================= 1. 核心引擎加载 (单例模式) =================
@st.cache_resource
def load_engine():
    """初始化 RAG 引擎，全局只执行一次"""
    print("正在初始化 RAG 引擎...")
    rag = MedicalRAG()
    return rag


# 加载引擎
with st.spinner("正在启动医疗核心引擎..."):
    engine = load_engine()

# ================= 2. 侧边栏：控制台 =================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/doctor-male--v1.png", width=60)
    st.title("控制台")

    # ------------------------------------------------
    # 模块 A: 病历场景选择
    # ------------------------------------------------
    st.markdown("### 📂 场景选择")

    # 读取 demo_cases.json
    cases = []
    try:
        if os.path.exists("demo_cases.json"):
            with open("demo_cases.json", "r", encoding="utf-8") as f:
                cases = json.load(f)
        else:
            st.warning("未找到 demo_cases.json")
    except Exception as e:
        st.error(f"加载病历库失败: {e}")

    # 提取选项
    case_names = [c["title"] for c in cases]
    options = ["-- 自定义输入 --"] + case_names

    # 下拉框
    selected_case_name = st.selectbox(
        "选择演示病历:",
        options,
        index=0
    )

    # 确定默认文本
    default_text = ""
    case_desc = "手动输入或粘贴文本"

    if selected_case_name != "-- 自定义输入 --":
        # 找到对应数据
        selected_data = next((c for c in cases if c["title"] == selected_case_name), None)
        if selected_data:
            default_text = selected_data.get("content", "")
            case_desc = selected_data.get("description", "")

    st.caption(f"当前场景：{case_desc}")

    st.divider()

    # ------------------------------------------------
    # 模块 B: 参数调优 (新增 Top-N)
    # ------------------------------------------------
    st.markdown("### 🎛️ 模型参数调优")
    with st.container():
        # 1. 阈值滑块
        new_threshold = st.slider(
            "相似度阈值 (Threshold)",
            min_value=0.0, max_value=1.0,
            value=engine.retrieval_threshold,
            step=0.05,
            help="向量检索初筛阈值。调低可增加召回（防止漏掉表格），调高更精准。"
        )

        # 2. 召回数量滑块
        new_k = st.slider(
            "初筛数量 (Retrieval K)",
            min_value=5, max_value=50,
            value=engine.retrieval_k,
            step=5,
            help="向量数据库初步召回的文档数量（建议设大一点，如30）。"
        )

        # 3. 重排数量滑块 (新增)
        new_rerank_n = st.slider(
            "重排数量 (Rerank Top-N)",
            min_value=1, max_value=15,
            value=getattr(engine, 'rerank_top_n', 5),  # 默认取值，防止属性不存在报错
            step=1,
            help="经 Reranker 精选后，最终喂给 LLM 的片段数量（建议 5-10）。"
        )

        # 实时应用配置到引擎实例
        if (new_threshold != engine.retrieval_threshold or
                new_k != engine.retrieval_k or
                new_rerank_n != engine.rerank_top_n):
            # 调用更新方法
            engine.update_config(k=new_k, threshold=new_threshold, kn=new_rerank_n)
            st.toast(f"参数更新: K={new_k}, Thr={new_threshold}, TopN={new_rerank_n}", icon="✅")

    st.divider()

    # ------------------------------------------------
    # 模块 C: 知识库管理
    # ------------------------------------------------
    st.markdown("### 📚 知识库管理")
    with st.expander("➕ 新增医学文档", expanded=False):
        uploaded_file = st.file_uploader("上传 TXT/MD 说明书", type=["txt", "md"])

        # 或者手动输入
        manual_text = st.text_area("或直接粘贴文本内容", height=100)
        manual_name = st.text_input("文档标题 (用于引用)", value="新补充说明书")

        if st.button("提交入库", use_container_width=True):
            content = ""
            source = ""

            if uploaded_file:
                content = uploaded_file.getvalue().decode("utf-8")
                source = uploaded_file.name
            elif manual_text:
                content = manual_text
                source = f"{manual_name}.txt"

            if content:
                with st.spinner("正在切分并写入向量库..."):
                    success = engine.add_knowledge(content, source)
                if success:
                    st.success(f"《{source}》已成功入库！")
                    time.sleep(1)
                    st.rerun()  # 刷新页面
            else:
                st.warning("请提供内容")

# ================= 3. 主界面：病历审核 =================

st.markdown('<div class="main-header">🏥 智能处方审核系统</div>', unsafe_allow_html=True)

# 显示当前生效的参数状态条 (更新显示 Top-N)
st.info(
    f"⚙️ 当前引擎配置：召回阈值 **{engine.retrieval_threshold}** | 初筛数量 **{engine.retrieval_k}** | 重排数量 **{engine.rerank_top_n}**")

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

            status_box = st.status("正在进行 AI 药学审查...", expanded=True)

            try:
                # 1. 结构化
                status_box.write("1. 正在结构化病历与意图识别...")

                # 2. 检索 (更新显示 Top-N)
                status_box.write(
                    f"2. 正在执行多路检索 (Thr={engine.retrieval_threshold}, K={engine.retrieval_k}, TopN={engine.rerank_top_n})...")

                # 3. 审核
                status_box.write("3. 正在生成决策...")

                start_time = time.time()
                # 调用核心逻辑
                result = engine.review_record(medical_input)
                end_time = time.time()

                status_box.update(label=f"审核完成 (耗时 {end_time - start_time:.2f}s)", state="complete",
                                  expanded=False)

                # === 展示结果 ===

                # 总结卡片
                summary = result.get("audit_report_summary", {})
                decision = summary.get("final_decision", "未知")
                color = "green"
                if "拦截" in decision:
                    color = "red"
                elif "人工" in decision or "慎用" in decision:
                    color = "orange"

                st.markdown(f"""
                <div style="padding: 15px; border-left: 5px solid {color}; background-color: #f9f9f9; border-radius: 5px;">
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
                    # 图标逻辑
                    ai_review = item.get('ai_review', '')
                    icon = "✅"
                    if any(x in ai_review for x in ["高", "禁忌", "拦截"]):
                        icon = "🔴"
                    elif any(x in ai_review for x in ["中", "慎用"]):
                        icon = "🟠"

                    with st.expander(f"{icon} 查询点：{item['query']}", expanded=True):
                        st.markdown(f"**AI 结论：**\n\n{ai_review}")

                        st.caption("📚 检索来源：")
                        sources = item.get('evidence_sources', [])
                        if sources:
                            for s in sources:
                                st.code(s, language=None)
                        else:
                            st.text("无相关来源")

                # 调试信息
                with st.expander("🛠️ 查看原始 JSON 响应 (Debug)"):
                    st.json(result)

            except Exception as e:
                st.error(f"发生错误: {e}")
                status_box.update(label="审核失败", state="error")
                st.exception(e)