import argparse
import copy
import os
import json
from typing import List
from tqdm import tqdm

try:
    from .debate4tran import Debate
    from .uncertainty_mi import compute_uncertainty_from_mad_runs
except ImportError:
    from debate4tran import Debate
    from uncertainty_mi import compute_uncertainty_from_mad_runs
from langcodes import Language

"""
外层控制器：
- 对 input 文件中的每个句子执行 K 次 MAD
- 收集 A1/B1
- 计算 MI (uncertainty)
- 保存结果到 uncertainty_results 中
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--apikey", type=str, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--min-runs", type=int, default=2, help="最少运行次数，达到后才检查不确定性阈值")
    parser.add_argument("--mi-threshold", type=float, default=0.05, help="互信息阈值，低于该值判定为低不确定性并提前停止")
    parser.add_argument("--sim-threshold", type=float, default=0.85, help="聚类相似度阈值")
    return parser.parse_args()


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

    src_lng, tgt_lng = lang_pair.split("-")

    os.makedirs(out_dir, exist_ok=True)

    config_template_path = os.path.join(os.path.dirname(__file__), "utils", "config4tran.json")
    config_template = json.load(open(config_template_path, "r", encoding="utf-8"))

    sentences = [line.strip() for line in open(input_file, "r", encoding="utf-8")]

    src_full = Language.make(language=src_lng).display_name()
    tgt_full = Language.make(language=tgt_lng).display_name()

    for idx, line in enumerate(tqdm(sentences, desc="Processing inputs")):
        parts = line.split(",")
        source = parts[0].strip()
        reference = parts[1].strip() if len(parts) > 1 else ""

        prompts_path = f"{out_dir}/tmp_config_{idx}.json"

        config = copy.deepcopy(config_template)
        config.update({
            "source": source,
            "reference": reference,
            "src_lng": src_full,
            "tgt_lng": tgt_full,
            "base_translation": ""
        })

        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        A1_list = []
        B1_list = []
        debate_outputs = []
        early_stop_reason = ""
        mi = 0.0
        a_clusters: List[int] = []
        b_clusters: List[int] = []

        # K 次 run
        for r in range(K):
            debate = Debate(
                save_file_dir=out_dir,
                num_players=3,
                openai_api_key=api_key,
                prompts_path=prompts_path,
                temperature=0,
                sleep_time=0
            )
            debate.run()

            # 提取 A1 / B1
            A1, B1 = debate.get_first_round_responses()

            A1_list.append(A1)
            B1_list.append(B1)

            debate_outputs.append(debate.save_file["debate_translation"])

            # 达到最少次数后，计算一次 MI 判断是否提前停止
            if len(A1_list) >= min_runs:
                mi, a_clusters, b_clusters = compute_uncertainty_from_mad_runs(
                    A1_list, B1_list, sim_threshold=sim_threshold
                )
                if mi <= mi_threshold:
                    early_stop_reason = (
                        f"Stopped after {len(A1_list)} runs because MI={mi:.4f} <= threshold {mi_threshold:.4f}"
                    )
                    break

        # 计算 uncertainty (MI)
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
            "final_outputs": debate_outputs,
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
        }

        json.dump(
            result,
            open(f"{out_dir}/result_{idx}.json", "w"),
            ensure_ascii=False,
            indent=4
        )

    print("All samples processed with uncertainty.")


if __name__ == "__main__":
    main()
