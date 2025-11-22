import argparse
import copy
import os
import json
from typing import List, Tuple, Dict, Any
from collections import Counter

from tqdm import tqdm
from langcodes import Language

try:
    from .debate4tran import Debate
    from .uncertainty_mi import compute_uncertainty_from_mad_runs
except ImportError:
    from debate4tran import Debate
    from uncertainty_mi import compute_uncertainty_from_mad_runs


"""
UA-MAD: Uncertainty-Aware Multi-Agent Debate

核心思路：
- 每个样本至少运行 1 次 MAD，把这一轮当作 baseline。
- 用 A1/B1 序列计算 MI（互信息），表示 epistemic uncertainty。
- 如果 MI 很低：说明模型对这句话比较“有把握”，不再跑更多 MAD，直接用第一次的翻译。
- 如果 MI 较高：针对该样本多跑几次 MAD（最多 max_runs 次），让模型在不确定的样本上多思考。
- 最后对多次 MAD 的输出做一次基于聚类/多数派的选优，得到 best_translation。
- 输出：
    - mi_uncertainty：最终 MI
    - best_translation：用于 COMET 评估的最终译文
    - all_translations：所有 MAD 轮次的译文
"""


def parse_args():
    parser = argparse.ArgumentParser("Uncertainty-Aware MAD Inference")

    parser.add_argument("--input", type=str, required=True,
                        help="Input file, one line: src,ref(optional)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output dir, will contain result_*.json")
    parser.add_argument("--lang", type=str, required=True,
                        help="Language pair, e.g. zh-en")
    parser.add_argument("--apikey", type=str, required=True,
                        help="OpenAI API key")

    # 不确定性相关超参
    parser.add_argument("--max-runs", type=int, default=3,
                        help="最大 MAD 轮数（同一句话最多跑几次完整 MAD）")
    parser.add_argument("--min-runs", type=int, default=1,
                        help="最少 MAD 次数（一般保持为 1 或 2）")
    parser.add_argument("--mi-low", type=float, default=0.03,
                        help="MI 低于该值认为模型很确定，可提前停止")
    parser.add_argument("--sim-threshold", type=float, default=0.85,
                        help="传给 compute_uncertainty_from_mad_runs 的聚类相似度阈值")

    # MAD 本身参数
    parser.add_argument("--model-name", type=str, default="gpt-3.5-turbo",
                        help="OpenAI model used in Debate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="MAD 里各代理的采样温度")
    parser.add_argument("--max-round", type=int, default=3,
                        help="每次 Debate 内部的最多辩论轮数")

    return parser.parse_args()


def load_config_template() -> Dict[str, Any]:
    """读取 MAD 原始项目中的 config4tran.json 作为模板"""
    this_dir = os.path.dirname(__file__)
    config_template_path = os.path.join(this_dir, "utils", "config4tran.json")
    with open(config_template_path, "r", encoding="utf-8") as f:
        config_template = json.load(f)
    return config_template


def safe_lang_name(tag: str) -> str:
    try:
        return Language.make(language=tag).display_name()
    except Exception:
        return tag


def run_single_mad(
    prompts_path: str,
    save_dir: str,
    api_key: str,
    model_name: str,
    temperature: float,
    max_round: int
) -> Tuple[str, str, str, Dict[str, Any]]:
    """
    运行一次完整 MAD：
    - 返回 A1, B1（第一轮正反方回答）
    - 返回 debate_translation（Moderator/Judge 给出的最终翻译）
    - 返回完整的 save_file（方便调试）
    """
    debate = Debate(
        model_name=model_name,
        temperature=temperature,
        num_players=3,
        save_file_dir=save_dir,
        openai_api_key=api_key,
        prompts_path=prompts_path,
        max_round=max_round,
        sleep_time=0,
    )
    debate.run()

    # 如果你在 Debate 类里实现了 get_first_round_responses，可以直接用；
    # 否则可以从 players 的 memory 里手动抽，我这里默认你已经有这个方法。
   # === NEW: check adaptive break ===
    adaptive_break = False
    if hasattr(debate, "is_adaptive_break"):
        adaptive_break = debate.is_adaptive_break()

    if hasattr(debate, "get_first_round_responses"):
        A1, B1 = debate.get_first_round_responses()
    else:
        A1, B1 = "", ""

    final_tran = debate.save_file.get("debate_translation", "")
    if not final_tran:
        try:
            final_tran = debate.save_file["players"]["Affirmative side"][-1]["content"]
        except:
            final_tran = ""

    return A1, B1, final_tran, debate.save_file, adaptive_break

def select_best_translation(
    translations: List[str],
    a_clusters: List[int],
    b_clusters: List[int]
) -> str:
    """
    给定多个 MAD 的翻译结果和 A/B 两方的聚类标签，
    选择一个“代表性最强”的翻译作为 best_translation。

    简单策略：
    1. 如果有 cluster 信息：
        - 用 A1_clusters + B1_clusters 统计全局出现最多的 cluster_id
        - 在这个主 cluster 对应的 run 里面选一个翻译作为代表
          （例如长度最短的那个，通常比较简洁）
    2. 如果 clusters 为空：
        - 退化为多数派投票（完全相同字符串的计数）
        - 如果仍然全部不同，就选第一条翻译。
    """

    # 统一清洗一下
    clean_tran = [t.strip() for t in translations if t and t.strip()]
    if not clean_tran:
        return ""  # 全部空，就返回空

    # 有 cluster 标签的情况
    if a_clusters and len(a_clusters) == len(clean_tran):
        # 把 A/B 的 cluster 都视作 signal，统计哪个 id 最常见
        all_cluster_ids = list(a_clusters) + list(b_clusters or [])
        main_cluster_id, _ = Counter(all_cluster_ids).most_common(1)[0]

        # 找到属于主 cluster 的翻译
        candidate_indices = [i for i, cid in enumerate(a_clusters) if cid == main_cluster_id]
        candidate_tran = [clean_tran[i] for i in candidate_indices if i < len(clean_tran)]

        if candidate_tran:
            # 选长度最短的一个（往往比较干净、去掉啰嗦解释）
            return min(candidate_tran, key=len)

    # 没有有效 clusters，或者长度不匹配 → 退化为多数派投票
    counter = Counter(clean_tran)
    best_tran, best_cnt = counter.most_common(1)[0]

    # 如果所有翻译都不同（best_cnt == 1），就直接返回第一条
    if best_cnt == 1:
        return clean_tran[0]

    return best_tran


def main():
    args = parse_args()

    input_file = args.input
    out_dir = args.output
    lang_pair = args.lang
    api_key = args.apikey

    max_runs = max(1, args.max_runs)
    min_runs = max(1, args.min_runs)
    mi_low = args.mi_low
    sim_threshold = args.sim_threshold

    model_name = args.model_name
    temperature = args.temperature
    max_round = args.max_round

    os.makedirs(out_dir, exist_ok=True)

    src_lng, tgt_lng = lang_pair.split("-")
    src_full = safe_lang_name(src_lng)
    tgt_full = safe_lang_name(tgt_lng)

    config_template = load_config_template()

    # 读取 input: 每一行 "src,ref(optional)"
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for idx, line in enumerate(tqdm(lines, desc="UA-MAD Inference")):
        parts = line.split(",")
        source = parts[0].strip()
        reference = parts[1].strip() if len(parts) > 1 else ""

        # 为当前样本生成 prompts config
        prompts_path = os.path.join(out_dir, f"tmp_config_{idx}.json")
        config = copy.deepcopy(config_template)
        config.update({
            "source": source,
            "reference": reference,
            "src_lng": src_full,
            "tgt_lng": tgt_full,
            "base_translation": ""  # 让 Debate 内部自动跑 baseline
        })
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        A1_list: List[str] = []
        B1_list: List[str] = []
        translations: List[str] = []  # 每次 MAD 的最终翻译
        mi = 0.0
        a_clusters: List[int] = []
        b_clusters: List[int] = []
        early_stop_reason = ""

        # === Step 1: 至少跑 min_runs 次 MAD ===
        for r in range(max_runs):
            A1, B1, tran, _, adaptive_break = run_single_mad(
                prompts_path=prompts_path,
                save_dir=out_dir,
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
                max_round=max_round,
            )
            A1_list.append(A1)
            B1_list.append(B1)
            translations.append(tran)

            if adaptive_break:
                early_stop_reason = (
                    f"Stopped after {len(A1_list)} runs due to adaptive break in Debate"
                )
                break

            # 满足最小运行次数后，才开始看 MI
            if len(A1_list) >= min_runs:
                mi, a_clusters, b_clusters = compute_uncertainty_from_mad_runs(
                    A1_list, B1_list, sim_threshold=sim_threshold
                )

                # 如果 MI 很低，说明模型对这句话比较“有信心”，无需额外 MAD
                if mi <= mi_low:
                    early_stop_reason = (
                        f"Stopped after {len(A1_list)} runs because MI={mi:.4f} <= mi_low={mi_low:.4f}"
                    )
                    break

            # 如果已经达到 max_runs，也会跳出循环
            if len(A1_list) >= max_runs:
                early_stop_reason = (
                    f"Reached max_runs={max_runs} with MI={mi:.4f}"
                )
                break

        # 保险再算一次 MI（统一记录）
        if A1_list:
            mi, a_clusters, b_clusters = compute_uncertainty_from_mad_runs(
                A1_list, B1_list, sim_threshold=sim_threshold
            )
        else:
            mi, a_clusters, b_clusters = 0.0, [], []
            early_stop_reason = "No successful MAD runs"

        # === 利用 cluster + 多轮 MAD 结果选一个 best_translation ===
        best_tran = select_best_translation(
            translations=translations,
            a_clusters=a_clusters,
            b_clusters=b_clusters,
        )

        result = {
            "id": idx,
            "source": source,
            "reference": reference,

            # 所有 MAD 轮次的翻译
            "all_translations": translations,

            # 用于评估的最终翻译（强烈建议 COMET 读取这个字段）
            "best_translation": best_tran,

            # 不确定性相关信息
            "mi_uncertainty": mi,
            "A1": A1_list,
            "B1": B1_list,
            "A1_clusters": a_clusters,
            "B1_clusters": b_clusters,

            # 运行信息
            "runs_completed": len(A1_list),
            "max_runs": max_runs,
            "min_runs": min_runs,
            "mi_low": mi_low,
            "sim_threshold": sim_threshold,
            "early_stop_reason": early_stop_reason,

            # 方便你复现实验
            "model_name": model_name,
            "temperature": temperature,
            "max_round": max_round,
            "src_lng": src_full,
            "tgt_lng": tgt_full,
        }

        out_path = os.path.join(out_dir, f"result_{idx}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    print("All samples processed with UA-MAD (uncertainty-aware MAD).")


if __name__ == "__main__":
    main()
