import os
import json
from comet import download_model, load_from_checkpoint
import torch

# ------- Correct Path -------
RESULT_DIR = "data/CommonMT/output"
SAVE_DIR = os.path.join(RESULT_DIR, "comet_scores_former")
os.makedirs(SAVE_DIR, exist_ok=True)    # <--- create folder
SAVE_PATH = os.path.join(SAVE_DIR, "comet_scores.json")

print("🔍 Using directory:", os.path.abspath(RESULT_DIR))

# ------- Debug: List files -------
all_files = os.listdir(RESULT_DIR)
print("📁 Files in directory:", all_files)

# ------- Load COMET model -------
print("🚀 Loading COMET model...")
model_path = download_model("wmt21-comet-qe-da")
model = load_from_checkpoint(model_path)

num_workers = 1 if torch.backends.mps.is_available() else 0

inputs = []
filenames = []

# ------- Load JSON files -------
for fname in sorted(all_files):
    if fname.endswith(".json") and fname[0].isdigit():
        fpath = os.path.join(RESULT_DIR, fname)

        print(f"📄 Reading {fname} ...")

        with open(fpath, "r") as f:
            data = json.load(f)

        src = data.get("source", "")
        mt  = data.get("debate_translation", "")
        ref = data.get("reference", "")

        if not mt or not ref:
            print(f"⚠️ Skipped {fname}: missing mt or reference")
            continue

        inputs.append({"src": src, "mt": mt, "ref": ref})
        filenames.append(fname)

print(f"📌 Total valid items collected: {len(inputs)}")

if len(inputs) == 0:
    print("❌ ERROR: No valid items found. Check directory path or JSON structure.")
    exit()

# ------- Run COMET -------
print("📊 Running COMET scoring...")
pred = model.predict(inputs, batch_size=8, gpus=0, num_workers=num_workers)["scores"]

# ------- Save -------
results = {"files": filenames, "scores": pred}

with open(SAVE_PATH, "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("✅ COMET scoring finished!")
print(f"💾 Saved to: {SAVE_PATH}")
