from typing import List, Optional

from transformers import AutoTokenizer, AutoModel


class ThudmChatGLM6B:
    model_name = "THUDM/chatglm-6b-int4"

    def __init__(self, pretrained_model_name_or_path: str):
        self.__tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path,
                                                         trust_remote_code=True)
        self.__model = AutoModel.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True).half().cuda()
        self.__history = []

    def chat(self, message: str):
        response, history = self.__model.chat(self.__tokenizer, message, history=self.__history)
        self.__history = history
        return response

    def new_session(self):
        self.__history = []
        return self


if __name__ == '__main__':
    homer = ThudmChatGLM6B('E:\\OneDrive\\Leqee\\ai\\THUDM\\chatglm-6b-int4')
    # response, history = model.chat(tokenizer, "你好", history=[])
    # print(response)
    # response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
    # print(response)

    while True:
        x = input('> ')
        if x == '':
            break
        r = homer.chat(x)
        print(f'< {r}')
