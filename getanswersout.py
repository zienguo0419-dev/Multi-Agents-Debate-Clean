import os
import json

# 设置 output 文件夹路径
folder_path = "./data/CommonMT/output"
output_txt = "./merged_translations.txt"

# 存储所有翻译结果
results = []

# 遍历文件夹
for filename in os.listdir(folder_path):

    # 要求是 .json
    if not filename.endswith(".json"):
        continue

    # 跳过 config 文件
    if "config" in filename.lower():
        continue

    # 文件名需是纯数字，例如 0.json、12.json
    name_no_ext = filename.replace(".json", "")
    if not name_no_ext.isdigit():
        continue

    file_id = int(name_no_ext)

    # 限制只取 0~19
    if not (0 <= file_id <= 19):
        continue

    # 开始读取有效文件
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

        supported_side = data.get("Supported Side", "").strip()
        base_translation = data.get("base_translation", "").strip()
        debate_translation = data.get("debate_translation", "").strip()

        # 根据 Supported Side 选择翻译
        if supported_side.lower().startswith("neg"):
            selected_translation = debate_translation or base_translation
        else:
            selected_translation = base_translation or debate_translation

        # 按文件 ID 存储，便于排序
        results.append((file_id, f"[{filename}] ({supported_side}) {selected_translation}"))

# 按 0~19 顺序排序
results.sort(key=lambda x: x[0])

# 输出到命令行
for _, text in results:
    print(text)

# 同时写入 merged_translations.txt
with open(output_txt, "w", encoding="utf-8") as out:
    out.write("\n".join(text for _, text in results))

print(f"\n✅ 已整合 {len(results)} 条翻译结果，输出到 {output_txt}")
