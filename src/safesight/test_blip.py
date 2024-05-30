import pathlib
import sys

import PIL.Image
import torch
from lavis.models import load_model_and_preprocess


def main(verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip_vqa", model_type="vqav2", is_eval=True, device=device
    )

    question = """Please answer with "yes" or "no". Is there a traffic accident in the image?"""

    positives = pathlib.Path("data/val/Accident")
    negatives = pathlib.Path("data/val/Non Accident")

    if not positives.exists() or not negatives.exists():
        print("Dataset not found.", file=sys.stderr)
        exit(-1)

    def test_dir(directory):
        yes, no = 0, 0

        for image in directory.iterdir():
            if verbose:
                print(f"Processing {image.name}...")
            raw_img = PIL.Image.open(image).convert("RGB")
            input_img = vis_processors["eval"](raw_img).unsqueeze(0).to(device)
            input_q = txt_processors["eval"](question)

            res = model.predict_answers(samples={"image": input_img, "text_input": input_q}, inference_method="generate")
            answer = res[0]

            if answer == "yes":
                if verbose:
                    print("Model answered 'yes'.")
                yes += 1
            elif answer == "no":
                if verbose:
                    print("Model answered 'no'.")
                no += 1
            # elif "yes" in answer:
            #     print(f"Model answered '{answer}', assuming 'yes'.")
            #     yes += 1
            # elif "no" in answer:
            #     print(f"Model answered '{answer}', assuming 'no'.")
            #     no += 1
            else:
                print(f"Model answered '{answer}', too ambiguous.")
        return yes, no

    true_positives, false_negatives = test_dir(positives)
    false_positives, true_negatives = test_dir(negatives)

    print("\n\n")
    print(f"""
    Test Results:
        * TP: {true_positives} # true positives
        * TN: {true_negatives} # true negatives
        * FP: {false_positives} # false positives
        * FN: {false_negatives} # false negatives
    """)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == "-v":
        main(True)
    else:
        main()
