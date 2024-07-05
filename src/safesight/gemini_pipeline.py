import google.generativeai as genai
from PIL.Image import Image

from safesight.pipeline import Pipeline, Evaluation

DEFAULT_SYSTEM_PROMPT = ("Your answer must start with either 'yes' or 'no',"
                         " then on the new line you must explain your answer.")

DEFAULT_PROMPT = ("Is there a car accident in this image? If you are not sure, you will answer 'no'."
                  " If you see the accident unclearly, you will say 'no'."
                  " After giving your verdict, explain your answer.")


class GeminiPipeline(Pipeline):
    def __init__(self, model_name="models/gemini-1.5-flash", system_prompt=DEFAULT_SYSTEM_PROMPT,
                 prompt=DEFAULT_PROMPT):
        self.model = genai.GenerativeModel(model_name=model_name, system_instruction=system_prompt)
        self.prompt = prompt

    def process_image(self, image: Image) -> Evaluation:
        response = self.model.generate_content([image, self.prompt])
        response.resolve()

        prediction = response.text.lower().startswith("yes")
        return Evaluation(prediction, response.text)

    def prepare(self):
        pass

    def cleanup(self):
        pass
