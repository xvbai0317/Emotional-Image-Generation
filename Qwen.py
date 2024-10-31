import random
import dashscope
import SD3
from http import HTTPStatus
from dashscope import Generation

# 复制api到这里
dashscope.api_key = ""


def call_with_messages():
    content = input("请输入你想问的问题：")
    messages = [{'role': 'system', 'content': 'Design a painting for me from the point of view of composition, color, elements, etc., '
                                              'combined with the emotions I give to your user, describe the content of the painting. '
                                              'Keep it to 77 words or less and must answer in English.'},
                {'role': 'user', 'content': content}]
# 自己选择模型可以
    response = Generation.call(model="qwen-turbo",
                               messages=messages,
                               # 设置随机数种子seed，如果没有设置，则随机数种子默认为1234
                               seed=random.randint(1, 10000),
                               temperature=0.8,
                               top_p=0.8,
                               top_k=50,
                               # 将输出设置为"message"格式
                               result_format='message')
    if response.status_code == HTTPStatus.OK:
        print("通义千问：" + response.output["choices"][0]["message"]["content"])
        SD3.generate_image(response.output["choices"][0]["message"]["content"], "./imgs/SD.png")
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))


if __name__ == '__main__':
    while True:
        call_with_messages()