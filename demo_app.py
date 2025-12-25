import streamlit as st
import json
import time
from rag_engine import MedicalRAG  # 引用你写好的引擎

# ================= 页面配置 =================
st.set_page_config(
    page_title="智能医疗辅助诊疗系统 RAG-DEMO",
    page_icon="🏥",
    layout="wide"
)

# ================= CSS 美化 =================
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6 }
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-medium { color: #ffa726; font-weight: bold; }
    .risk-safe { color: #00c853; font-weight: bold; }
    .audit-box { border: 1px solid #e0e0e0; padding: 15px; border-radius: 5px; background: white; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)


# ================= 1. 模型加载 (带缓存) =================
@st.cache_resource
def load_engine():
    """
    使用 cache_resource 装饰器，确保模型只加载一次，
    切换病历时不会重新初始化 RAG 引擎。
    """
    print("正在初始化 RAG 引擎...")
    # 这里初始化你的类
    rag = MedicalRAG()
    return rag


# 侧边栏：加载状态
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/doctor-male--v1.png", width=80)
    st.title("AI 药师工作台")

    with st.spinner("正在启动医疗核心引擎 (Loading Models)..."):
        engine = load_engine()
    st.success("✅ 引擎就绪 (Models Loaded)")

    st.markdown("---")
    st.markdown("### ⚙️ 演示控制")

    # 读取用例库
    try:
        with open("demo_cases.json", "r", encoding="utf-8") as f:
            cases = json.load(f)
    except FileNotFoundError:
        st.error("未找到 demo_cases.json，请检查文件位置")
        cases = []

    # 下拉选择框
    case_names = [c["title"] for c in cases]
    selected_case_name = st.selectbox("选择演示病历", ["-- 自定义输入 --"] + case_names)

# ================= 2. 主界面逻辑 =================

st.header("🏥 智能处方审核系统 (RAG-Audit)")

# 获取当前选中的病历内容
if selected_case_name == "-- 自定义输入 --":
    default_text = ""
    case_desc = "手动输入测试数据"
else:
    # 找到对应的病历数据
    selected_data = next(c for c in cases if c["title"] == selected_case_name)
    default_text = selected_data["content"]
    case_desc = selected_data["description"]

# 展示两栏布局：左边输入，右边结果
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 诊疗单/病历输入")
    st.info(f"当前场景：{case_desc}")

    medical_input = st.text_area(
        "病历文本 (支持手动修改)",
        value=default_text,
        height=300,
        help="模拟从 HIS 系统读取的非结构化文本"
    )

    audit_btn = st.button("🚀 开始智能审核", type="primary", use_container_width=True)

# ================= 3. 审核执行与展示 =================

if audit_btn and medical_input:
    with col2:
        st.subheader("🔍 审核报告")

        # 进度条模拟
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # --- 第一步：结构化与拆解 ---
            status_text.text("1/3 正在进行病历结构化与意图识别...")
            progress_bar.progress(30)

            # 调用你的核心方法
            result = engine.review_record(medical_input)

            # --- 第二步：原子查询展示 (这是 RAG 的亮点，要展示出来) ---
            status_text.text("2/3 正在执行多路混合检索 (Hybrid Retrieval)...")
            progress_bar.progress(60)

            with st.expander("🧠 AI 思维链 (原子查询拆解)", expanded=True):
                if "audit_logic_trace" in result:
                    for q in result["audit_logic_trace"]:
                        st.markdown(f"- 🔎 `{q}`")
                else:
                    st.write("未生成查询拆解")

            # --- 第三步：渲染最终结果 ---
            status_text.text("3/3 生成最终决策报告...")
            progress_bar.progress(100)
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()

            # 获取总结部分
            summary = result.get("audit_report_summary", {})
            details = result.get("audit_report_details", [])
            # 如果 summary 意外地变成了字符串（比如错误信息），将其转化为字典
            if isinstance(summary, str):
                summary = {
                    "final_decision": "系统异常",
                    "summary_text": summary,  # 把错误字符串放这里
                    "actionable_advice": "请检查后台日志"
                }

            # 1. 顶部大卡片：最终决策
            decision = summary.get("final_decision", "未知")
            color = "green"
            if "拦截" in decision:
                color = "red"
            elif "人工" in decision or "慎用" in decision:
                color = "orange"

            st.markdown(f"""
            <div style="padding: 20px; background-color: {'#ffebee' if color == 'red' else '#e8f5e9'}; border-radius: 10px; border-left: 5px solid {color};">
                <h3 style="margin:0; color:{color}">🛡️ 最终决策：{decision}</h3>
                <p style="margin-top:10px"><b>综合评价：</b>{summary.get("summary_text", "无")}</p>
                <p><b>建议操作：</b>{summary.get("actionable_advice", "无")}</p>
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            # 2. 详情列表
            st.markdown("#### 🧾 风险详情分析")
            for item in details:
                # 解析 AI 回复的风险等级
                ai_review = item.get('ai_review', '')
                risk_icon = "✅"
                if "高" in ai_review:
                    risk_icon = "🔴"
                elif "中" in ai_review:
                    risk_icon = "🟠"
                elif "低" in ai_review:
                    risk_icon = "🟡"

                with st.container():
                    st.markdown(f"""
                    <div class="audit-box">
                        <div style="font-size: 0.9em; color: gray;">针对查询：{item['query']}</div>
                        <div style="font-size: 1.1em; margin: 5px 0;">{risk_icon} <b>AI 结论：</b>{ai_review}</div>
                        <div style="font-size: 0.8em; color: #666;">📚 证据来源：{', '.join(item.get('evidence_sources', []))}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # 3. 原始 JSON (方便调试或展示数据结构)
            with st.expander("查看原始 JSON 数据"):
                st.json(result)

        except Exception as e:
            st.error(f"审核过程中发生错误: {str(e)}")
            st.exception(e)

elif audit_btn and not medical_input:
    st.warning("请输入病历文本或在左侧选择演示案例。")