from __future__ import annotations
from openai import OpenAI
import backoff
import time
import json
from typing import List, Dict, Optional
from openai import (
    APIStatusError, APIConnectionError, 
    RateLimitError, OpenAIError
)
from .openai_utils import (
    OutOfQuotaException,
    AccessTerminatedException,
    num_tokens_from_string,
    model2max_context
)

# 你支持的模型
SUPPORT_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4"
]

class Agent:
    """
    一个强健、可控、可 debug 的 Agent，用于 Multi-Agent Debate + MI Analysis。
    """

    def __init__(
        self,
        model_name: str,
        name: str,
        temperature: float,
        sleep_time: float = 0,
        api_key: Optional[str] = None,
        max_memory: int = 40
    ) -> None:
        assert model_name in SUPPORT_MODELS, f"Model {model_name} not in {SUPPORT_MODELS}"

        self.model_name = model_name
        self.name = name
        self.temperature = temperature
        self.sleep_time = sleep_time
        self.openai_api_key = api_key  # 必须设置，否则 ask 会报错
        self.memory_lst: List[Dict] = []
        self.max_memory = max_memory

        # 用于 MI
        self.trace_outputs = []   # append all raw assistant outputs
        self.trace_messages = []  # append messages before each forward

    # ----------------------------
    #       核心 LLM 调用
    # ----------------------------
    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIStatusError, OpenAIError, APIConnectionError),
        max_tries=10
    )
    def query(
        self, 
        messages: List[Dict], 
        max_tokens: int,
        api_key: str,
        temperature: float
    ) -> str:
        """真正发送请求到 OpenAI API"""
        time.sleep(self.sleep_time)

        print(f"\n[DEBUG] Sending request → {self.model_name}")
        print(f"[DEBUG] Context messages = {len(messages)}")

        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,  # 新 SDK
                timeout=30,
            )
            content = response.choices[0].message.content
            print("[DEBUG] Response received.")
            return content

        except RateLimitError as e:
            print(f"[RATE LIMIT ERROR] {e}")
            if "quota" in str(e).lower():
                raise OutOfQuotaException(api_key)
            elif "terminated" in str(e).lower():
                raise AccessTerminatedException(api_key)
            else:
                raise e
        
        except Exception as e:
            print(f"[ERROR] LLM Query Failed: {e}")
            raise e

    # ----------------------------
    #       Context 管理
    # ----------------------------
    def set_meta_prompt(self, meta_prompt: str):
        """设置 system prompt。自动避免重复添加多个 system prompt。"""
        # remove old system prompts
        self.memory_lst = [m for m in self.memory_lst if m["role"] != "system"]
        self.memory_lst.insert(0, {"role": "system", "content": meta_prompt})

    def add_event(self, event: str):
        """添加 user 的输入事件"""
        self.memory_lst.append({"role": "user", "content": event})

    def add_memory(self, memory: str):
        """添加 assistant 输出（或你自己的人工注入）"""
        self.memory_lst.append({"role": "assistant", "content": memory})
        print(f"\n----- [{self.name} output] -----\n{memory}\n")

        # 记录用于 MI 分析
        self.trace_outputs.append(memory)

    # 自动截断过长上下文（保留 system + 最近 max_memory 条）
    def _truncate_memory(self):
        if len(self.memory_lst) <= self.max_memory:
            return
        sys_prompt = [m for m in self.memory_lst if m["role"] == "system"]
        other_msgs = [m for m in self.memory_lst if m["role"] != "system"]
        self.memory_lst = sys_prompt + other_msgs[-self.max_memory:]

    # ----------------------------
    #       ask()：主接口
    # ----------------------------
    def ask(self, temperature: Optional[float] = None) -> str:
        """执行一次模型调用，用当前 memory 生成下一个 assistant 消息"""

        # 截断上下文
        self._truncate_memory()

        # 计算剩余 token
        num_context_tokens = sum(
            num_tokens_from_string(m["content"], self.model_name)
            for m in self.memory_lst
        )
        max_token = model2max_context[self.model_name] - num_context_tokens
        max_token = max(128, max_token)  # 最低给点空间

        # 记录 trace
        self.trace_messages.append(list(self.memory_lst))

        # 发送请求
        output = self.query(
            messages=self.memory_lst,
            max_tokens=max_token,
            api_key=self.openai_api_key,
            temperature=temperature or self.temperature
        )

        # 保存输出
        self.add_memory(output)
        return output

    # ----------------------------
    #       JSON 结果校验
    # ----------------------------
    def ask_json(self, temperature: Optional[float] = None) -> Dict:
        """
        要求模型返回 JSON 格式。如果不是合法 JSON，会自动 retry 1 次。
        非常适合 moderator / judge。
        """

        text = self.ask(temperature=temperature)

        try:
            return json.loads(text)
        except Exception:
            print("\n[WARN] JSON parsing failed. Trying second attempt...")
            # 强制要求它只输出 JSON
            self.add_event("⚠️ Your previous output was not valid JSON. Output STRICT JSON ONLY.")
            text2 = self.ask(temperature=temperature)
            try:
                return json.loads(text2)
            except Exception as e:
                print(f"[ERROR] JSON retry still failed: {e}")
                raise e
