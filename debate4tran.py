"""
MAD: Multi-Agent Debate with Large Language Models
Copyright (C) 2023  The MAD Team

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import json
import random
# random.seed(0)
import argparse
from langcodes import Language
try:
    from .utils.agent import Agent
except ImportError:
    from utils.agent import Agent
from datetime import datetime
from tqdm import tqdm


NAME_LIST = [
    "Affirmative side",
    "Negative side",
    "Moderator",
]


class DebatePlayer(Agent):
    def __init__(self, model_name: str, name: str, temperature: float, openai_api_key: str, sleep_time: float) -> None:
        """Create a player in the debate

        Args:
            model_name(str): model name
            name (str): name of this player
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            openai_api_key (str): As the parameter name suggests
            sleep_time (float): sleep because of rate limits
        """
        super(DebatePlayer, self).__init__(model_name, name, temperature, sleep_time)
        # 新版 Agent 里有 openai_api_key 参数，但这里保持向后兼容：直接赋值属性即可
        self.openai_api_key = openai_api_key


class Debate:
    def __init__(self,
                 model_name: str = 'gpt-3.5-turbo',
                 temperature: float = 0,
                 num_players: int = 3,
                 save_file_dir: str = None,
                 openai_api_key: str = None,
                 prompts_path: str = None,
                 max_round: int = 3,
                 sleep_time: float = 0
                 ) -> None:
        """Create a debate

        Args:
            model_name (str): openai model name
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            num_players (int): num of players
            save_file_dir (str): dir path to json file
            openai_api_key (str): As the parameter name suggests
            prompts_path (str): prompts path (json file)
            max_round (int): maximum Rounds of Debate
            sleep_time (float): sleep because of rate limits
        """

        self.model_name = model_name
        self.temperature = temperature
        self.num_players = num_players
        self.save_file_dir = save_file_dir
        self.openai_api_key = openai_api_key
        self.max_round = max_round
        self.sleep_time = sleep_time
        self.first_round_affirmative = ""
        self.first_round_negative = ""

        # init save file
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
        self.save_file = {
            'start_time': current_time,
            'end_time': '',
            'model_name': model_name,
            'temperature': temperature,
            'num_players': num_players,
            'success': False,
            "src_lng": "",
            "tgt_lng": "",
            'Ch': '',
            'EngT': '',
            'EngF': '',
            'base_translation': '',
            "debate_translation": '',
            "Reason": '',
            "Supported Side": '',
            'players': {},
        }
        prompts = json.load(open(prompts_path, "r"))
        self.save_file.update(prompts)
        self.init_prompt()

        if self.save_file['base_translation'] == "":
            self.create_base()

        # create & init agents
        self.creat_agents()
        self.init_agents()

    def init_prompt(self):
        def prompt_replace(key):
            self.save_file[key] = (
                self.save_file[key]
                .replace("##src_lng##", self.save_file["src_lng"])
                .replace("##tgt_lng##", self.save_file["tgt_lng"])
                .replace("##source##", self.save_file["source"])
                .replace("##base_translation##", self.save_file["base_translation"])
            )

        # 这些 prompt 里包含 src/tgt/source/base_translation 占位符
        prompt_replace("base_prompt")
        prompt_replace("player_meta_prompt")
        prompt_replace("moderator_meta_prompt")
        prompt_replace("judge_prompt_last2")

    def create_base(self):
        print(f"\n===== Translation Task =====\n{self.save_file['base_prompt']}\n")
        agent = DebatePlayer(
            model_name=self.model_name,
            name='Baseline',
            temperature=self.temperature,
            openai_api_key=self.openai_api_key,
            sleep_time=self.sleep_time,
        )
        agent.add_event(self.save_file['base_prompt'])
        base_translation = agent.ask()  # ask() 已经会自动 add_memory
        self.save_file['base_translation'] = base_translation
        self.save_file['affirmative_prompt'] = self.save_file['affirmative_prompt'].replace(
            "##base_translation##", base_translation
        )
        self.save_file['players'][agent.name] = agent.memory_lst

    def creat_agents(self):
        # creates players
        self.players = [
            DebatePlayer(
                model_name=self.model_name,
                name=name,
                temperature=self.temperature,
                openai_api_key=self.openai_api_key,
                sleep_time=self.sleep_time
            )
            for name in NAME_LIST
        ]
        self.affirmative = self.players[0]
        self.negative = self.players[1]
        self.moderator = self.players[2]

    def init_agents(self):
        # start: set meta prompt
        self.affirmative.set_meta_prompt(self.save_file['player_meta_prompt'])
        self.negative.set_meta_prompt(self.save_file['player_meta_prompt'])
        self.moderator.set_meta_prompt(self.save_file['moderator_meta_prompt'])

        # start: first round debate, state opinions
        print(f"===== Debate Round-1 =====\n")

        # Affirmative first statement
        self.affirmative.add_event(self.save_file['affirmative_prompt'])
        self.aff_ans = self.affirmative.ask()  # ask 已自动保存到 memory

        # Negative response
        neg_prompt = self.save_file['negative_prompt'].replace('##aff_ans##', self.aff_ans)
        self.negative.add_event(neg_prompt)
        self.neg_ans = self.negative.ask()

        # Moderator judgment (JSON)
        mod_prompt = (
            self.save_file['moderator_prompt']
            .replace('##aff_ans##', self.aff_ans)
            .replace('##neg_ans##', self.neg_ans)
            .replace('##round##', 'first')
        )
        self.moderator.add_event(mod_prompt)
        # 使用 ask_json 强制 JSON 输出
        self.mod_ans = self.moderator.ask_json()

        # 记录首轮发言，便于外部脚本使用
        self.first_round_affirmative = self.aff_ans
        self.first_round_negative = self.neg_ans
        self.save_file['first_round_affirmative'] = self.first_round_affirmative
        self.save_file['first_round_negative'] = self.first_round_negative

    def round_dct(self, num: int):
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth',
            6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return dct[num]

    def save_file_to_json(self, id):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
        save_file_path = os.path.join(self.save_file_dir, f"{id}.json")

        self.save_file['end_time'] = current_time
        json_str = json.dumps(self.save_file, ensure_ascii=False, indent=4)
        with open(save_file_path, 'w') as f:
            f.write(json_str)

    def broadcast(self, msg: str):
        """Broadcast a message to all players.

        Args:
            msg (str): the message
        """
        for player in self.players:
            player.add_event(msg)

    def speak(self, speaker: str, msg: str):
        """The speaker broadcast a message to all other players.

        Args:
            speaker (str): name of the speaker
            msg (str): the message
        """
        if not msg.startswith(f"{speaker}: "):
            msg = f"{speaker}: {msg}"
        for player in self.players:
            if player.name != speaker:
                player.add_event(msg)

    def ask_and_speak(self, player: DebatePlayer):
        ans = player.ask()
        self.speak(player.name, ans)

    def run(self):

        for round_idx in range(self.max_round - 1):

            if self.mod_ans.get("debate_translation", "") != '':
                break
            else:
                print(f"===== Debate Round-{round_idx + 2} =====\n")

                # Affirmative refutes negative
                self.affirmative.add_event(
                    self.save_file['debate_prompt'].replace('##oppo_ans##', self.neg_ans)
                )
                self.aff_ans = self.affirmative.ask()

                # Negative refutes affirmative
                self.negative.add_event(
                    self.save_file['debate_prompt'].replace('##oppo_ans##', self.aff_ans)
                )
                self.neg_ans = self.negative.ask()

                # Moderator judges this round
                mod_prompt = (
                    self.save_file['moderator_prompt']
                    .replace('##aff_ans##', self.aff_ans)
                    .replace('##neg_ans##', self.neg_ans)
                    .replace('##round##', self.round_dct(round_idx + 2))
                )
                self.moderator.add_event(mod_prompt)
                self.mod_ans = self.moderator.ask_json()

        # 如果 moderator 已经给出最终翻译
        if self.mod_ans.get("debate_translation", "") != '':
            self.save_file.update(self.mod_ans)
            self.save_file['success'] = True

        # ultimate deadly technique: fallback judge
        else:
            judge_player = DebatePlayer(
                model_name=self.model_name,
                name='Judge',
                temperature=self.temperature,
                openai_api_key=self.openai_api_key,
                sleep_time=self.sleep_time
            )

            # 这里仍然取第一次 round 的发言（避免后面复杂历史干扰）
            aff_ans = self.affirmative.memory_lst[2]['content']
            neg_ans = self.negative.memory_lst[2]['content']

            judge_player.set_meta_prompt(self.save_file['moderator_meta_prompt'])

            # extract answer candidates（不强制 JSON）
            judge_player.add_event(
                self.save_file['judge_prompt_last1']
                .replace('##aff_ans##', aff_ans)
                .replace('##neg_ans##', neg_ans)
            )
            _ = judge_player.ask()  # 只为让模型列出候选

            # select one from the candidates（JSON 格式）
            judge_player.add_event(self.save_file['judge_prompt_last2'])
            ans_json = judge_player.ask_json()

            if ans_json.get("debate_translation", "") != '':
                self.save_file['success'] = True

            self.save_file.update(ans_json)
            self.players.append(judge_player)

        # 记录所有玩家的对话
        for player in self.players:
            self.save_file['players'][player.name] = player.memory_lst

    def get_first_round_responses(self):
        """
        返回首轮正反双方的回答，避免外部脚本直接依赖 memory 的内部结构
        """
        return self.first_round_affirmative, self.first_round_negative
    def is_adaptive_break(self):
        """If the debate_translation is produced early, debate ended early."""
        try:
            return bool(self.save_file.get("debate_translation", "").strip())
        except:
            return False


def parse_args():
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input-file", type=str, required=True, help="Input file path")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Output file dir")
    parser.add_argument("-lp", "--lang-pair", type=str, required=True, help="Language pair")
    parser.add_argument("-k", "--api-key", type=str, required=True, help="OpenAI api key")
    parser.add_argument("-m", "--model-name", type=str, default="gpt-3.5-turbo", help="Model name")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Sampling temperature")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    openai_api_key = args.api_key

    current_script_path = os.path.abspath(__file__)
    MAD_path = current_script_path.rsplit("/", 2)[0]

    src_lng, tgt_lng = args.lang_pair.split('-')
    src_full = Language.make(language=src_lng).display_name()
    tgt_full = Language.make(language=tgt_lng).display_name()

    config = json.load(open(f"{MAD_path}/code/utils/config4tran.json", "r"))

    inputs = open(args.input_file, "r").readlines()
    inputs = [l.strip() for l in inputs]

    save_file_dir = args.output_dir
    if not os.path.exists(save_file_dir):
        os.mkdir(save_file_dir)

    for id, line in enumerate(tqdm(inputs)):
        # 原始格式：source,reference
        parts = line.split(',')
        if len(parts) < 2:
            # 简单防御：如果格式不对，就跳过或只用 source
            src_text = parts[0]
            ref_text = ""
        else:
            src_text = parts[0]
            ref_text = parts[1]

        prompts_path = f"{save_file_dir}/{id}-config.json"

        config['source'] = src_text
        config['reference'] = ref_text
        config['src_lng'] = src_full
        config['tgt_lng'] = tgt_full

        with open(prompts_path, 'w') as file:
            json.dump(config, file, ensure_ascii=False, indent=4)

        debate = Debate(
            save_file_dir=save_file_dir,
            num_players=3,
            openai_api_key=openai_api_key,
            prompts_path=prompts_path,
            temperature=args.temperature,
            sleep_time=0,
            model_name=args.model_name
        )
        debate.run()
        debate.save_file_to_json(id)
