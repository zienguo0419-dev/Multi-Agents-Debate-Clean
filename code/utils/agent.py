from openai import OpenAI
import backoff
import time
from openai import APIStatusError, APIConnectionError, RateLimitError, OpenAIError
from .openai_utils import OutOfQuotaException, AccessTerminatedException
from .openai_utils import num_tokens_from_string, model2max_context

support_models = ['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4o', 'gpt-4']

class Agent:
    def __init__(self, model_name: str, name: str, temperature: float, sleep_time: float=0) -> None:
        self.model_name = model_name
        self.name = name
        self.temperature = temperature
        self.memory_lst = []
        self.sleep_time = sleep_time

    @backoff.on_exception(backoff.expo, (RateLimitError, APIStatusError, OpenAIError, APIConnectionError), max_tries=10)
    def query(self, messages: "list[dict]", max_tokens: int, api_key: str, temperature: float) -> str:
        """make a query to OpenAI API"""
        time.sleep(self.sleep_time)
        assert self.model_name in support_models, f"Not support {self.model_name}. Choices: {support_models}"
        print(f"[DEBUG] Sending request to {self.model_name} ...")

        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,  # new SDK uses max_completion_tokens
                timeout=30  # prevent infinite waiting
            )
            gen = response.choices[0].message.content
            print("[DEBUG] Got response.")
            return gen

        except RateLimitError as e:
            print(f"[ERROR] Rate limit hit: {e}")
            if "quota" in str(e):
                raise OutOfQuotaException(api_key)
            elif "terminated" in str(e):
                raise AccessTerminatedException(api_key)
            else:
                raise e
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            raise e

    def set_meta_prompt(self, meta_prompt: str):
        self.memory_lst.append({"role": "system", "content": f"{meta_prompt}"})

    def add_event(self, event: str):
        self.memory_lst.append({"role": "user", "content": f"{event}"})

    def add_memory(self, memory: str):
        self.memory_lst.append({"role": "assistant", "content": f"{memory}"})
        print(f"----- {self.name} -----\n{memory}\n")

    def ask(self, temperature: float=None):
        num_context_token = sum([num_tokens_from_string(m["content"], self.model_name) for m in self.memory_lst])
        max_token = model2max_context[self.model_name] - num_context_token
        return self.query(self.memory_lst, max_token, api_key=self.openai_api_key, temperature=temperature or self.temperature)
