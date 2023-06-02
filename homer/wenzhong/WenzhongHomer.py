import time

from transformers import pipeline, set_seed


class WenzhongHomer:
    model_name = 'IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese'

    def __init__(self, model, **kwargs):
        """
        :param model: 'IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese'
        :param kwargs:
        """
        self.__pipeline = pipeline('text-generation', model=model)

    def set_seed_of_pipeline(self, seed):
        set_seed(seed)

    def generate_results(self, prompt: str, **kwargs):
        """

        :param prompt:
        :param kwargs:
            max_length=30, num_return_sequences=5
        :return:
        """
        return self.__pipeline(prompt, **kwargs)


if __name__ == '__main__':
    model_path = 'E:\\sinri\\homer\\models\\Wenzhong2.0-GPT2-3.5B-chinese'
    print(f'{time.time()} 开始加载模型...')
    homer = WenzhongHomer(model_path, device="cuda")
    print(f'{time.time()} 开始调用模型...')
    results = homer.generate_results(
        '获取信用卡的方式',
        max_length=1024,
        num_return_sequences=1
    )
    print(f'{time.time()} 开始渲染返回结果...')
    for result in results:
        print("answer:")
        print(result['generated_text'])
