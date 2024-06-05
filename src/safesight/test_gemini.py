import os
import pathlib
import sys

import PIL.Image
import click
import google.generativeai as genai

from safesight.model_api import TestResults
from safesight.cli import cli

# GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
# genai.configure(api_key=GOOGLE_API_KEY)


@cli.group()
def gemini():
    """Commands for Gemini (python>=3.9)"""
    pass


@gemini.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument("question", type=str)
@click.option("-v", "--verbose", is_flag=True, help="click.echo verbose output")
def test_dir(path, question, verbose):
    """ Test the Gemini model on a directory of images. Outputs the number of 'yes' and 'no' answers.
        Use the GOOGLE_API_KEY environment variable to set the API key."""

    if os.environ.get("GOOGLE_API_KEY") is None:
        click.echo("GOOGLE_API_KEY environment variable isn't set.", file=sys.stderr)
        exit(-1)

    results = {}
    model = genai.GenerativeModel("gemini-pro-vision")
    # positives = "data/val/Accident"
    # negatives = "data/val/Non Accident"

    # false_positives = 0
    # false_negatives = 0
    # total = 0
    positives, negatives = 0, 0

    for file in list(pathlib.Path(path).iterdir()):
        click.echo(f"Querying {file.name}...")

        img = PIL.Image.open(file)

        response = model.generate_content(
            [
                img,
                f"You can only answer with 'Yes' or 'No'. {question}",
            ]
        )
        response.resolve()

        if "yes" in response.text.lower():
            prediction = True
        elif "no" in response.text.lower():
            prediction = False
        else:
            prediction = None

        if verbose and response.text.lower() not in ["yes", "no"]:
            click.echo(f"Gemini's response: {response.text}.")
            if prediction is None:
                click.echo("Could not deduce the response.")
            else:
                click.echo(f"Assuming prediction: {prediction}.")

        results[file.name] = (True, prediction)

        if prediction is False:
            negatives += 1
        if prediction is True:
            positives += 1

    click.echo()

    # click.echo(f"False positives: {false_positives}.")
    # click.echo(f"False negatives: {false_negatives}.")
    # click.echo(f"Total: {total}.")
    # click.echo(f"Accuracy: {(total - false_positives - false_negatives) / total * 100}%.")
    #
    # return TestResults(
    #     [label for label, _ in results.values()],
    #     [result for _, result in results.values()],
    # )

    click.echo(f"Positives: {positives}.")
    click.echo(f"Negatives: {negatives}.")