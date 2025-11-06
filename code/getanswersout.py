import os
import json

# 设置 output 文件夹路径
folder_path = "./data/CommonMT/output"
output_txt = "./merged_translations.txt"

# 存储所有翻译结果
results = []

for filename in os.listdir(folder_path):
    # 跳过文件名中包含 "config" 的 JSON 文件
    if filename.endswith(".json") and "config" not in filename:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            # 提取支持方和对应翻译
            supported_side = data.get("Supported Side", "").strip()
            base_translation = data.get("base_translation", "").strip()
            debate_translation = data.get("debate_translation", "").strip()

            # 根据 Supported Side 判断要选哪一个翻译
            if supported_side.lower().startswith("neg"):
                selected_translation = debate_translation or base_translation
            else:
                selected_translation = base_translation or debate_translation

            # 添加到结果列表
            results.append(f"[{filename}] ({supported_side}) {selected_translation}")

# 输出到命令行
for r in results:
    print(r)

# 同时写入到一个文件
with open(output_txt, "w", encoding="utf-8") as out:
    out.write("\n".join(results))

print(f"\n✅ 已整合 {len(results)} 条翻译结果，输出到 {output_txt}")