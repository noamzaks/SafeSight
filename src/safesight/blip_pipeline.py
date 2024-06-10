from safesight.pipeline import Pipeline, Evaluation
import torch
from lavis.models import load_model_and_preprocess
from typing import Optional
from PIL.Image import Image
import PIL.Image


class BlipPipeline(Pipeline):
    def __init__(self, question: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.vis_processors, self.txt_processors = (
            load_model_and_preprocess(
                name="blip_vqa", model_type="vqav2", is_eval=True, device=self.device
            )
        )

        self.last_evaluation = Evaluation(False)

        self.question = f"""Please answer with "yes" or "no". {question}"""

    def process_image(self, image: Image) -> Optional[Evaluation]:
        input_img = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        input_q = self.txt_processors["eval"](self.question)

        res = self.model.predict_answers(
            samples={"image": input_img, "text_input": input_q},
            inference_method="generate",
        )
        answer = res[0]

        if answer == "yes":
            self.last_evaluation = Evaluation(True)
        self.last_evaluation = Evaluation(False)

        return self.last_evaluation

    def evaluate(self) -> Evaluation:
        return self.last_evaluation


if __name__ == "__main__":
    blip = BlipPipeline("Is there a road accident in this photo?")
    print(blip.process_image(PIL.Image.open("data/val/Accident/test24_41.jpg")))
