import os
import csv
import json
from comet import download_model, load_from_checkpoint

# ===== PATHS =====
RESULT_DIR = "./data/CommonMT/output/uncertainty_results"
CSV_PATH = "./data/CommonMT/contextless syntactic ambiguity.csv"
SAVE_PATH = "./data/CommonMT/output/comet_scores_contrastive_0_19.json"

MODEL_NAME = "Unbabel/wmt22-comet-da"
FILE_IDS = list(range(0, 20))  # ⭐⭐⭐ 只评 0~19
# =====================

print("🚀 Loading COMET model...")
model_path = download_model(MODEL_NAME)
model = load_from_checkpoint(model_path)

def comet_score(src, mt, ref):
    data = [{"src": src, "mt": mt, "ref": ref}]
    scores = model.predict(data, batch_size=1, gpus=0, num_workers=1)
    return float(scores["scores"][0])

# ===== 1. Load CSV =====
csv_rows = []
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if len(row) < 3:
            continue
        src = row[0].strip()
        ref = row[1].strip()
        neg = row[2].strip()
        csv_rows.append({"id": i, "src": src, "ref": ref, "neg": neg})

print(f"✔ Loaded {len(csv_rows)} rows from CSV.")

# ===== 2. Evaluate (only id in 0–19) =====
summary = {}
correct_discrimination = 0

for row in csv_rows:
    idx = row["id"]

    if idx not in FILE_IDS:  # ⭐ 只处理 0~19
        continue

    src = row["src"]
    ref = row["ref"]
    neg = row["neg"]

    json_path = os.path.join(RESULT_DIR, f"result_{idx}.json")
    if not os.path.isfile(json_path):
        print(f"⚠ Missing {json_path}, skip.")
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 取你的模型翻译结果
    hyp = data.get("best_translation")
    if not hyp:
        outputs = data.get("final_outputs", [])
        hyp = outputs[0] if outputs else ""

    # COMET: correct vs negative
    score_pos = comet_score(src, hyp, ref)
    score_neg = comet_score(src, hyp, neg)

    discrim = score_pos > score_neg
    if discrim:
        correct_discrimination += 1

    summary[idx] = {
        "source": src,
        "reference": ref,
        "negative_ref": neg,
        "hypothesis": hyp,
        "score_pos": score_pos,
        "score_neg": score_neg,
        "discriminates_correctly": discrim
    }

total = len(summary)
acc = correct_discrimination / total if total > 0 else 0

print("\n====================================")
print(f"🎯 Correct discrimination in 0–19: {correct_discrimination}/{total} = {acc:.3f}")
print("====================================")

with open(SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=4)

print(f"📁 Saved: {SAVE_PATH}")
