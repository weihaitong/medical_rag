# main.py
from rag_engine import MedicalRAG

def main():
    print("正在初始化医疗 RAG 系统...")
    try:
        rag = MedicalRAG(data_path="data/")

        print("\n 系统就绪！请输入您的医疗问题（输入 'quit' 退出）\n")

        while True:
            question = input("🩺 您的问题: ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                print("感谢使用医疗辅助系统，再见！")
                break
            if not question:
                continue

            try:
                answer = rag.ask(question)
                print(f"\n💡 回答:\n{answer}\n")
            except Exception as e:
                print(f" 回答生成出错: {e}\n")
    except Exception as e:
        print(f" 系统初始化失败: {e}")
        print("请检查：")
        print("1. data/ 目录下是否有医学文档")
        print("2. 网络连接是否正常（首次运行需要下载模型）")
        print("3. 依赖包是否正确安装")

if __name__ == "__main__":
    main()