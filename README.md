# SafeSight
Detecting Suspicious Events in Video.

# Installation

1. Clone the repository
2. Install rye
3. Run ``` rye sync ```

# Usage

1. Build the repository (``` rye build ```)
2. Activate python virtual environment, if not already (``` source .venv/bin/activate ```)
3. To download the dataset, run ``` python -m safesight.dataset_downloader ```
4. Unpack the dataset (``` unzip -x dataset.zip ```)
5. To test Gemini's performance, run ``` python -m safesight.test_gemini ```. ``` GOOGLE_API_KEY ``` environment variable must be set to a valid Google API key.
