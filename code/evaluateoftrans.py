import re
import numpy as np
from openai import OpenAI
from PyPDF2 import PdfReader

client = OpenAI(api_key="")  # ← 请换成你自己的 key

# 1️⃣ 读取 PDF 内容
pdf_path = "/Users/guozien/Desktop/gptturbo3.5mad.pdf"
reader = PdfReader(pdf_path)
text = "\n".join(page.extract_text() for page in reader.pages)

# 2️⃣ 拆分 GPT-4 翻译和参考答案
gpt_text = re.search(r"Gpt-3.5-turbo with MAD的答案：(.*?)参考答案：", text, re.S).group(1).strip()
ref_text = re.search(r"参考答案：(.*)", text, re.S).group(1).strip()

# 3️⃣ 分句（去除空行）
def split_sentences(block):
    lines = [l.strip().strip("，。,.") for l in block.split("\n") if l.strip()]
    # 去除中文行，只留英文
    return [l for l in lines if re.search(r"[a-zA-Z]", l)]

gpt_lines = split_sentences(gpt_text)
ref_lines = split_sentences(ref_text)
pairs = list(zip(gpt_lines, ref_lines))
print(f"共检测到 {len(pairs)} 对句子。")

# 4️⃣ 计算相似度
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

threshold = 0.85
correct = 0

for i, (pred, ref) in enumerate(pairs, 1):
    emb_pred = client.embeddings.create(input=pred, model="text-embedding-3-small").data[0].embedding
    emb_ref = client.embeddings.create(input=ref, model="text-embedding-3-small").data[0].embedding
    sim = cosine_similarity(np.array(emb_pred), np.array(emb_ref))
    if sim >= threshold:
        correct += 1

# 5️⃣ 输出正确率
accuracy = correct / len(pairs)
print(f"\n📊 GPT-4 翻译语义正确率：{accuracy:.2%}")
