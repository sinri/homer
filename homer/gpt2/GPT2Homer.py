import transformers
from transformers import pipeline, TextGenerationPipeline, set_seed


class GPT2Homer:
    def __init__(self, model, **kwargs):
        # self.__pipeline = TextGenerationPipeline(model, **kwargs)
        self.__pipeline = pipeline('text-generation', model=model, **kwargs)

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
    model_path = 'E:\\sinri\\homer\\models\\gpt2'
    homer = GPT2Homer(model_path)
    results = homer.generate_results(
        'how to apply for a credit card?',
        max_length=1024,
        num_return_sequences=2
    )
    for result in results:
        print("answer:")
        print(result['generated_text'])
