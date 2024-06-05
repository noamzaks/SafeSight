import pathlib
import sys

import PIL.Image
import click
import torch
from lavis.models import load_model_and_preprocess

from safesight.cli import cli


@cli.group()
def lavis():
    """Commands for the LAVIS library (BLIP model) (python==3.8)"""
    pass


@lavis.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument("question", type=str)
@click.option("-v", "--verbose", is_flag=True, help="click.echo verbose output")
def test_dir(path, question, verbose):
    """ Test the BLIP model on a directory of images. Outputs the number of 'yes' and 'no' answers. """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip_vqa", model_type="vqav2", is_eval=True, device=device
    )

    question = f"""Please answer with "yes" or "no". {question}"""

    # positives = pathlib.Path("data/val/Accident")
    # negatives = pathlib.Path("data/val/Non Accident")

    # if not positives.exists() or not negatives.exists():
    #     click.echo("Dataset not found.", file=sys.stderr)
    #     exit(-1)

    def test_dir(directory: pathlib.Path):
        yes, no = 0, 0

        for image in directory.iterdir():
            if verbose:
                click.echo(f"Processing {image.name}...")
            raw_img = PIL.Image.open(image).convert("RGB")
            input_img = vis_processors["eval"](raw_img).unsqueeze(0).to(device)
            input_q = txt_processors["eval"](question)

            res = model.predict_answers(samples={"image": input_img, "text_input": input_q},
                                        inference_method="generate")
            answer = res[0]

            if answer == "yes":
                if verbose:
                    click.echo("Model answered 'yes'.")
                yes += 1
            elif answer == "no":
                if verbose:
                    click.echo("Model answered 'no'.")
                no += 1
            # elif "yes" in answer:
            #     click.echo(f"Model answered '{answer}', assuming 'yes'.")
            #     yes += 1
            # elif "no" in answer:
            #     click.echo(f"Model answered '{answer}', assuming 'no'.")
            #     no += 1
            else:
                click.echo(f"Model answered '{answer}', too ambiguous.")
        return yes, no

    # true_positives, false_negatives = test_dir(positives)
    # false_positives, true_negatives = test_dir(negatives)

    positives, negatives = test_dir(pathlib.Path(path))

    click.echo("\n\n")
    # click.echo(f"""
    # Test Results:
    #     * TP: {true_positives} # true positives
    #     * TN: {true_negatives} # true negatives
    #     * FP: {false_positives} # false positives
    #     * FN: {false_negatives} # false negatives
    # """)
    click.echo(f"""
    Test Results:
        * Positives: {positives}
        * Negatives: {negatives}
    """)
