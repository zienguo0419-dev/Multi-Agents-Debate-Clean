from openai import OpenAI
import os

# 初始化客户端
client = OpenAI(api_key="#YOUR_OPENAI_API_KEY#")  # 替换为你的 OpenAI API 密钥
messages = [
    {"role": "system", "content": "You are a professional translator proficient in Chinese and English."},
    {"role": "user", "content":"""Given the Chinese sentence 
吃掉敌人一个师。
     
"""}
]

# 运行三轮，只保留第三轮结果
final_output = None

for i in range(3):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 或 "gpt-4o"
        messages=messages,
    )
    result = response.choices[0].message.content

    # 更新对话上下文
    messages.append({"role": "assistant", "content": result})

    # 第三轮时保存输出
    if i == 2:
        final_output = result

# 打印第三轮的输出
print(final_output)
