# main.py
import logging
import os

if os.path.exists("./qdrant_db/qdrant.lock"):
    os.remove("./qdrant_db/qdrant.lock")

def setup_logging(log_file="logs/medical_rag.log", level=logging.INFO):
    """统一日志配置：同时输出到控制台和文件"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 避免重复添加 handler
    if root_logger.handlers:
        return

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def main():
    logging.info("正在初始化医疗 RAG 系统...")
    try:
        from rag_engine import MedicalRAG  # 延迟导入（可选，非必须）
        rag = MedicalRAG(data_path="data/")

        logging.info("系统就绪！请输入您的医疗问题（输入 'quit' 退出）")

        while True:
            question = input("🩺 您的问题: ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                logging.info("感谢使用医疗辅助系统，再见！")
                break
            if not question:
                continue

            try:
                answer = rag.ask(question)
                logging.info(f"💡 回答:\n{answer}")
            except Exception as e:
                logging.error(f"回答生成出错: {e}", exc_info=True)

    except Exception as e:
        logging.critical(f"系统初始化失败: {e}", exc_info=True)
        logging.info("请检查：")
        logging.info("1. data/ 目录下是否有医学文档")
        logging.info("2. 网络连接是否正常（首次运行需要下载模型）")
        logging.info("3. 依赖包是否正确安装")


if __name__ == "__main__":
    setup_logging()  # 先配置日志
    main()