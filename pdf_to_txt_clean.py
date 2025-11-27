# pdf_to_txt_clean.py
import os
import re
import pdfplumber
from pathlib import Path


def clean_text(text: str) -> str:
    """
    清理 PDF 提取的文本：
    - 去除页眉/页脚（如期刊名、页码）
    - 合并断行（保留段落）
    - 去除多余空行和空白
    """
    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # 跳过明显是页眉/页脚的行（可根据实际文档调整规则）
        if not line:
            continue
        if re.match(r"^\d+$", line):  # 纯数字页码
            continue
        if any(header in line for header in [
            "中华", "医学", "杂志", "指南", "国家", "卫健委",
            "第.*卷", "第.*期", "www", "©", "ISSN", "CN"
        ]) and len(line) < 30:
            continue

        # 合并断行：如果行末是中文字符（非标点），则与下一行合并
        if cleaned_lines and re.search(r"[\u4e00-\u9fff]$", cleaned_lines[-1]) and not re.search(r"[。！？；]$|[\dA-Za-z]$",
                                                                                                 cleaned_lines[-1]):
            cleaned_lines[-1] += line
        else:
            cleaned_lines.append(line)

    # 合并段落（保留空行分隔）
    result = "\n\n".join([line for line in cleaned_lines if line.strip()])

    # 进一步清理：多个换行 → 双换行，多余空格
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r" {2,}", " ", result)

    return result.strip()


def convert_pdf_to_txt(pdf_path: str, output_dir: str = "data/"):
    """
    将单个 PDF 转为干净的 TXT 文件
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    txt_filename = pdf_path.stem + ".txt"
    txt_path = output_dir / txt_filename

    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                # 提取文本（保留换行）
                text = page.extract_text(
                    layout=False,  # 不严格保留布局（避免空格过多）
                    x_tolerance=1,  # 容忍微小水平偏移
                    y_tolerance=1
                )
                if text:
                    full_text += text + "\n\n"

        # 清理文本
        clean = clean_text(full_text)

        # 写入 TXT
        with open(txt_path, "w", encoding="utf-8") as f:
            # 可选：在文件开头添加来源注释
            source_comment = f"[来源：{pdf_path.name}]\n\n"
            f.write(source_comment + clean)

        print(f"✅ 已转换：{pdf_path.name} → {txt_path}")

    except Exception as e:
        print(f"❌ 转换失败 {pdf_path.name}: {e}")


def batch_convert_pdf_to_txt(pdf_dir: str, output_dir: str = "data/"):
    """
    批量转换目录下所有 PDF
    """
    pdf_dir = Path(pdf_dir)
    for pdf_file in pdf_dir.glob("*.pdf"):
        convert_pdf_to_txt(pdf_file, output_dir)


# =============================
# 使用示例
# =============================
if __name__ == "__main__":
    # 方式1：转换单个 PDF
    # convert_pdf_to_txt("downloads/diabetes_guide_2022.pdf")

    # 方式2：批量转换整个目录
    batch_convert_pdf_to_txt(pdf_dir="downloads/", output_dir="data/")

    print("\n🎉 所有 PDF 转换完成！请检查 data/ 目录。")