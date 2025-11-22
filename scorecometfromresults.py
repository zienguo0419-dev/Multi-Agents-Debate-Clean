import os
import json
from comet import download_model, load_from_checkpoint

# -------- CONFIG -------
RESULT_DIR = "./data/CommonMT/output/uncertainty_results"
SAVE_DIR = "./data/CommonMT/output/comet_scores"
# NOTE: COMET models hosted on Hugging Face require the full repo id.
MODEL_NAME = "Unbabel/wmt22-comet-da"
FILE_IDS = list(range(0, 199))  # 只读取 result_0.json ~ result_19.json
# -----------------------

os.makedirs(SAVE_DIR, exist_ok=True)

print("🚀 Loading COMET model...")
model_path = download_model(MODEL_NAME)
model = load_from_checkpoint(model_path)

def comet_score_one(source, hypothesis, reference):
    """给定 source / hypothesis / reference，返回单句 COMET 分数"""
    data = [{"src": source, "mt": hypothesis, "ref": reference}]
    scores = model.predict(data, batch_size=1, gpus=0, num_workers=1)
    return scores["scores"][0]


print("📊 Start scoring results...")

# -------- 改进：只保留指定 ID 范围内的 result_{id}.json 或 id.json --------
all_files = []
for idx in FILE_IDS:
    candidates = [f"result_{idx}.json", f"{idx}.json"]
    found = None
    for fname in candidates:
        fpath = os.path.join(RESULT_DIR, fname)
        if os.path.isfile(fpath):
            found = fname
            break
    if found:
        all_files.append(found)
    else:
        print(f"⚠ Missing files for ID={idx}, skip.")

print(f"📁 Found {len(all_files)} result JSON files.")

summary = {}

for filename in all_files:
    file_path = os.path.join(RESULT_DIR, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_id = data.get("id")
    if file_id is None:
        digits = "".join(ch for ch in filename if ch.isdigit())
        file_id = int(digits) if digits else -1

    # -------- 提取数据 --------
    source = data.get("source", "").strip()
    reference = data.get("reference", "").replace("，", "").strip()

    # 没有 final_outputs 则跳过
    if "final_outputs" not in data:
        print(f"⚠ Skipping {filename} (missing final_outputs)")
        continue

    outputs = data["final_outputs"]

    print(f"\n➡ Scoring {filename} (ID={file_id}), translations = {len(outputs)}")

    sample_scores = []

    for i, hyp in enumerate(outputs):
        score = comet_score_one(source, hyp, reference)
        print(f"  - Output {i}: {score:.4f}")
        sample_scores.append(score)

    mean_score = sum(sample_scores) / len(sample_scores)

    summary[file_id] = {
        "source": source,
        "reference": reference,
        "final_outputs": outputs,
        "scores": sample_scores,
        "mean_score": mean_score
    }

# 保存总文件
save_path = os.path.join(SAVE_DIR, "comet_scores.json")
json.dump(summary, open(save_path, "w"), ensure_ascii=False, indent=4)

print("\n============================================")
print("✔ All COMET scoring completed!")
print(f"✔ Results saved to: {save_path}")
print("============================================")
