# demo_app.py
import streamlit as st
import time
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

# ================= 2. 侧边栏：动态配置与数据管理 =================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/doctor-male--v1.png", width=60)
    st.title("控制台")

    # --- 模块 A: 参数调优 ---
    st.markdown("### 🎛️ 模型参数调优")
    with st.container():
        # 1. 阈值滑块
        new_threshold = st.slider(
            "相似度阈值 (Threshold)",
            min_value=0.0, max_value=1.0,
            value=engine.retrieval_threshold,  # 读取当前内存中的值
            step=0.05,
            help="低于此分数的文档将被初筛过滤。调低可增加召回，调高更精准。"
        )

        # 2. 召回数量滑块
        new_k = st.slider(
            "召回数量 (Top-K)",
            min_value=1, max_value=30,
            value=engine.retrieval_k,
            step=1,
            help="向量检索阶段初筛的文档数量。"
        )

        # 实时应用配置到引擎实例
        if new_threshold != engine.retrieval_threshold or new_k != engine.retrieval_k:
            engine.update_config(k=new_k, threshold=new_threshold)
            st.toast(f"参数已更新: K={new_k}, Thr={new_threshold}", icon="✅")

    st.divider()

    # --- 模块 B: 知识库管理 ---
    st.markdown("### 📚 知识库管理")
    with st.expander("➕ 新增医学文档", expanded=False):
        uploaded_file = st.file_uploader("上传 TXT/MD 说明书", type=["txt", "md"])

        # 或者手动输入
        manual_text = st.text_area("或直接粘贴文本内容", height=100)
        manual_name = st.text_input("文档标题 (用于引用)", value="新补充说明书")

        if st.button("提交入库"):
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

# 显示当前生效的参数
st.info(f"当前生效参数：召回阈值 **{engine.retrieval_threshold}** | 召回数量 **{engine.retrieval_k}**")

col1, col2 = st.columns([1, 1])

# 预设病历
demo_text = """患者：李小宝，男，4岁，体重16kg。
主诉：发热1天，体温39℃。
诊断：上呼吸道感染。
处方：
1. 布洛芬混悬液 10ml po qid
2. 左氧氟沙星片 0.1g po bid"""

with col1:
    st.subheader("📋 病历输入")
    medical_input = st.text_area(
        "病历文本",
        value=demo_text,
        height=300
    )
    audit_btn = st.button("🚀 开始智能审核", type="primary", use_container_width=True)

if audit_btn and medical_input:
    with col2:
        st.subheader("🔍 审核报告")

        status_box = st.status("正在进行 AI 药学审查...", expanded=True)

        try:
            # 1. 结构化
            status_box.write("1. 正在结构化病历...")
            # 2. 检索
            status_box.write("2. 正在执行多路检索 (使用当前侧边栏参数)...")
            # 3. 审核
            status_box.write("3. 正在生成决策...")

            start_time = time.time()
            result = engine.review_record(medical_input)
            end_time = time.time()

            status_box.update(label=f"审核完成 (耗时 {end_time - start_time:.2f}s)", state="complete", expanded=False)

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
                <p style="margin-top:10px;">{summary.get("summary_text", "")}</p>
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            # 详情与证据
            st.markdown("#### 🔬 风险详情与证据链")
            details = result.get("audit_report_details", [])

            if not details:
                st.info("未发现具体的风险点或未触发检索。")

            for item in details:
                with st.expander(f"💊 查询点：{item['query']}", expanded=True):
                    st.markdown(f"**AI 结论：** {item['ai_review']}")

                    # 只有在 debug 模式下看具体的 evidence
                    st.caption("📚 检索到的支持证据 (Top 3)：")
                    # 这里我们需要把 engine 检索过程中的 evidence 传递出来
                    # 在目前的 review_record 返回结果中，details 里的 evidence_sources 只是文件名
                    # 如果要看具体文本，需要在 rag_engine 的 _execute_batch_audit 里把文本也存进去
                    # 咱们目前代码里存的是 sources list，这里展示文件名即可
                    st.code(f"来源文件: {item.get('evidence_sources', [])}")

            # 调试信息
            with st.expander("查看原始 JSON 响应"):
                st.json(result)

        except Exception as e:
            st.error(f"发生错误: {e}")
            status_box.update(label="审核失败", state="error")