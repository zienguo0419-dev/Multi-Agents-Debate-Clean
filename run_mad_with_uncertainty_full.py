import argparse
import copy
import os
import json
from typing import List, Dict, Any
from tqdm import tqdm
import importlib.resources as importlib_resources

# Python 3.8 compatibility: backfill importlib.resources.files via importlib_resources
if not hasattr(importlib_resources, "files"):
    import importlib_resources as _importlib_resources
    import sys
    sys.modules["importlib.resources"] = _importlib_resources

try:
    from .debate4tran import Debate
    from .uncertainty_mi import compute_uncertainty_from_mad_runs
except ImportError:
    from debate4tran import Debate
    from uncertainty_mi import compute_uncertainty_from_mad_runs

from langcodes import Language


def parse_args():
    parser = argparse.ArgumentParser("MAD + Uncertainty (full version)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input file, one line per sample: src,ref")
    parser.add_argument("--output", type=str, required=True,
                        help="Directory to save result_*.json")
    parser.add_argument("--lang", type=str, required=True,
                        help="Language pair, e.g. zh-en")
    parser.add_argument("--apikey", type=str, required=True,
                        help="OpenAI API key")
    parser.add_argument("--runs", type=int, default=5,
                        help="Max MAD runs per sample")
    parser.add_argument("--min-runs", type=int, default=2,
                        help="Min runs before checking MI")
    parser.add_argument("--mi-threshold", type=float, default=0.05,
                        help="Mutual information threshold for early stop")
    parser.add_argument("--sim-threshold", type=float, default=0.85,
                        help="Clustering similarity threshold")
    parser.add_argument("--model-name", type=str, default="gpt-3.5-turbo",
                        help="LLM backbone")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for debaters/judge")
    return parser.parse_args()


def load_config_template() -> Dict[str, Any]:
    # 假设和原项目一样，config4tran.json 在 code/utils 下
    this_dir = os.path.dirname(__file__)
    config_template_path = os.path.join(this_dir, "utils", "config4tran.json")
    with open(config_template_path, "r", encoding="utf-8") as f:
        config_template = json.load(f)
    return config_template


def main():
    args = parse_args()

    input_file = args.input
    out_dir = args.output
    lang_pair = args.lang
    api_key = args.apikey
    K = args.runs
    min_runs = max(1, args.min_runs)
    mi_threshold = args.mi_threshold
    sim_threshold = args.sim_threshold

    model_name = args.model_name
    temperature = args.temperature

    os.makedirs(out_dir, exist_ok=True)

    src_lng, tgt_lng = lang_pair.split("-")
    src_full = Language.make(language=src_lng).display_name()
    tgt_full = Language.make(language=tgt_lng).display_name()

    config_template = load_config_template()

    # 读取输入数据：每一行 "src,ref"
    with open(input_file, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    for idx, line in enumerate(tqdm(sentences, desc="Processing inputs")):
        parts = line.split(",")
        source = parts[0].strip()
        reference = parts[1].strip() if len(parts) > 1 else ""

        # 为当前样本生成 prompts 配置文件
        prompts_path = os.path.join(out_dir, f"tmp_config_{idx}.json")
        config = copy.deepcopy(config_template)
        config.update({
            "source": source,
            "reference": reference,
            "src_lng": src_full,
            "tgt_lng": tgt_full,
            "base_translation": ""   # 让 Debate 内部自动跑 baseline
        })
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        A1_list: List[str] = []
        B1_list: List[str] = []
        final_translations: List[str] = []
        debate_jsons: List[dict] = []

        early_stop_reason = ""
        mi = 0.0
        a_clusters: List[int] = []
        b_clusters: List[int] = []

        # 对单个样本进行最多 K 次 MAD
        for r in range(K):
            debate = Debate(
                model_name=model_name,
                temperature=temperature,
                num_players=3,
                save_file_dir=out_dir,
                openai_api_key=api_key,
                prompts_path=prompts_path,
                max_round=3,
                sleep_time=0
            )
            debate.run()

            # 保存完整的 debate.json（如果你想调试用）
            debate_jsons.append(copy.deepcopy(debate.save_file))

            # 从你自己扩展过的 Debate 类拿 first-round outputs
            # 如果没有 get_first_round_responses，你可以从 memory_lst 里自己抽
            if hasattr(debate, "get_first_round_responses"):
                A1, B1 = debate.get_first_round_responses()
            else:
                # fallback: 不定义 A1/B1（你可以按自己项目来改）
                A1, B1 = "", ""

            A1_list.append(A1)
            B1_list.append(B1)

            # 使用 moderator/judge 给出的最终翻译
            final_tran = debate.save_file.get("debate_translation", "")
            if not final_tran:
                # 如果 Moderator/Judge 没产出，就用 Affirmative 的最后一次答复兜底
                final_tran = debate.save_file["players"]["Affirmative side"][-1]["content"]
            final_translations.append(final_tran)

            # 达到最少运行次数后，计算 MI 并判断是否提前停止
            if len(A1_list) >= min_runs:
                mi, a_clusters, b_clusters = compute_uncertainty_from_mad_runs(
                    A1_list, B1_list, sim_threshold=sim_threshold
                )
                if mi <= mi_threshold:
                    early_stop_reason = (
                        f"Stopped after {len(A1_list)} runs because MI={mi:.4f} <= threshold {mi_threshold:.4f}"
                    )
                    break

        # 保险再算一次 MI（统一记录）
        if A1_list:
            mi, a_clusters, b_clusters = compute_uncertainty_from_mad_runs(
                A1_list, B1_list, sim_threshold=sim_threshold
            )
        else:
            mi, a_clusters, b_clusters = 0.0, [], []
            early_stop_reason = "No successful debate runs"

        result = {
            "id": idx,
            "source": source,
            "reference": reference,
            "final_translations": final_translations,   # 每次 MAD 的最终译文
            "A1": A1_list,
            "B1": B1_list,
            "A1_clusters": a_clusters,
            "B1_clusters": b_clusters,
            "mi_uncertainty": mi,
            "runs_requested": K,
            "runs_completed": len(A1_list),
            "min_runs": min_runs,
            "mi_threshold": mi_threshold,
            "sim_threshold": sim_threshold,
            "early_stop_reason": early_stop_reason,
            # 如果你想保存所有 debate 详细过程，可以保留这行；否则可以注释掉
            # "all_debate_json": debate_jsons,
        }

        out_path = os.path.join(out_dir, f"result_{idx}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    print("All samples processed with MAD + uncertainty (full version).")


if __name__ == "__main__":
    main()
