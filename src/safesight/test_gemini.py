import os
import pathlib

import PIL.Image
import google.generativeai as genai

# GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
# genai.configure(api_key=GOOGLE_API_KEY)


def main():
    model = genai.GenerativeModel('gemini-pro-vision')
    positives = 'data/val/Accident'
    negatives = 'data/val/Non Accident'

    false_positives = 0
    false_negatives = 0
    total = 0

    for file in pathlib.Path(positives).iterdir():
        print(f"Querying {file.name}...")

        img = PIL.Image.open(file)

        response = model.generate_content(
            [img,
             "You can only answer with 'Yes' or 'No'. Is there an accident in this image?"]
        )
        response.resolve()

        if 'yes' in response.text.lower():
            prediction = True
        elif 'no' in response.text.lower():
            prediction = False
        else:
            prediction = None

        if response.text.lower() not in ['yes', 'no']:
            print(f"Gemini's response: {response.text}.")
            if prediction is None:
                print("Could not deduce the response.")
            else:
                print(f"Assuming prediction: {prediction}.")

        if prediction is False:
            false_negatives += 1
        if prediction is not None:
            total += 1

    for file in pathlib.Path(negatives).iterdir():
        print(f"Querying {file.name}...")

        img = PIL.Image.open(file)

        response = model.generate_content(
            [img,
             "You can only answer with 'Yes' or 'No'. Is there an accident in this image?"]
        )
        response.resolve()

        if 'yes' in response.text.lower():
            prediction = True
        elif 'no' in response.text.lower():
            prediction = False
        else:
            prediction = None

        if response.text.lower() not in ['yes', 'no']:
            print(f"Gemini's response: {response.text}.")
            if prediction is None:
                print("Could not deduce the response.")
            else:
                print(f"Assuming prediction: {prediction}.")

        if prediction is True:
            false_positives += 1
        if prediction is not None:
            total += 1

    print()

    print(f"False positives: {false_positives}.")
    print(f"False negatives: {false_negatives}.")
    print(f"Total: {total}.")
    print(f"Accuracy: {(total - false_positives - false_negatives) / total * 100}%.")


if __name__ == '__main__':
    main()
