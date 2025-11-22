import argparse
import os
import json
from typing import List, Dict, Any
from tqdm import tqdm

try:
    from .utils.agent import Agent
except ImportError:
    from utils.agent import Agent

from langcodes import Language


# 这里写一个非常轻量级的“不确定性”计算：
# 简单思路：翻译结果去重后的个数越多，认为不确定性越高
# 你也可以换成你自己的 compute_uncertainty_* 函数
def compute_simple_uncertainty(translations: List[str]) -> float:
    if not translations:
        return 0.0
    unique = set(t.strip() for t in translations if t.strip())
    # 归一化到 [0,1]：1 - (max_count / K)
    # 如果都一样 => 不确定性接近 0
    # 如果全都不同 => 不确定性接近 1
    K = len(translations)
    max_count = max(translations.count(u) for u in unique)
    return 1.0 - (max_count / K)


def parse_args():
    parser = argparse.ArgumentParser("Single-Agent Translation + Uncertainty (fast version)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input file, one line: src,ref(optional)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output dir for result_*.json")
    parser.add_argument("--lang", type=str, required=True,
                        help="Language pair, e.g. zh-en")
    parser.add_argument("--apikey", type=str, required=True,
                        help="OpenAI API key")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of samples per sentence")
    parser.add_argument("--model-name", type=str, default="gpt-3.5-turbo",
                        help="Backbone LLM")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (建议>0 生成多样翻译)")
    return parser.parse_args()


def main():
    args = parse_args()

    input_file = args.input
    out_dir = args.output
    lang_pair = args.lang
    api_key = args.apikey
    K = max(1, args.runs)

    model_name = args.model_name
    temperature = args.temperature

    os.makedirs(out_dir, exist_ok=True)

    src_lng, tgt_lng = lang_pair.split("-")
    src_full = Language.make(language=src_lng).display_name()
    tgt_full = Language.make(language=tgt_lng).display_name()

    # 构造一个简单的翻译 prompt，你可以按照自己 config4tran 里的 base_prompt 来对齐
    base_prompt_template = (
        "You are a professional translator. Translate the following sentence "
        "from {src_lng} to {tgt_lng}.\n"
        "Source: {source}\n"
        "Translation:"
    )

    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for idx, line in enumerate(tqdm(lines, desc="Fast single-agent translation")):
        parts = line.split(",")
        source = parts[0].strip()
        reference = parts[1].strip() if len(parts) > 1 else ""

        translations: List[str] = []

        for r in range(K):
            prompt = base_prompt_template.format(
                src_lng=src_full,
                tgt_lng=tgt_full,
                source=source,
            )
            agent = Agent(
                model_name=model_name,
                name=f"Translator_{r}",
                temperature=temperature,
                sleep_time=0,
            )
            agent.openai_api_key = api_key  # 如果你的 Agent 类用这个属性
            agent.add_event(prompt)
            tran = agent.ask()
            translations.append(tran)

        # 计算一个非常便宜的“不确定性”
        unc = compute_simple_uncertainty(translations)

        result = {
            "id": idx,
            "source": source,
            "reference": reference,
            "translations": translations,
            "simple_uncertainty": unc,
            "runs": K,
            "model_name": model_name,
            "temperature": temperature,
            "src_lng": src_full,
            "tgt_lng": tgt_full,
        }

        out_path = os.path.join(out_dir, f"result_{idx}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    print("All samples processed with single-agent fast uncertainty.")


if __name__ == "__main__":
    main()
