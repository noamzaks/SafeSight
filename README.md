# SafeSight
Detecting Suspicious Events in Video.

# Installation

1. Clone the repository ``` git clone https://github.com/noamzaks/SafeSight.git ```
2. Install [uv](https://github.com/astral-sh/uv/tree/main). \
   (Linux/macOS: ``` curl -LsSf https://astral.sh/uv/install.sh | sh ```, Windows: ``` powershell -c "irm https://astral.sh/uv/install.ps1 | iex" ```, for other methods check the repository).
   Make sure the uv executable is in your PATH (its environment files are sourced).
3. Create a python virtual environment with the correct python version: \
   ``` uv venv -p <VERSION> [PATH] ``` \
   For that, corresponding python version should be installed on the system.

   Current constraints:
   - mlcroissant (for downloading datasets) will only be installed on `python>=3.10`
   - LAVIS (BLIP model) will only be installed on `python==3.8.*`
   - Gemini API will only be installed on `python>=3.9`

   You can have multiple virtual environments with different python versions in different paths. The default path is ".venv", and uv will override the virtual environment if it already exists.
4. Activate the virtual environment with the correct python version: \
   ``` source .venv/bin/activate ``` \
   (for Windows: ``` .\.venv\Scripts\Activate.ps1 ```)
5. Install the current package with its dependencies: \
   ``` uv pip install -e . ```

   ### Pytorch notice:
   By default, CUDA version of Pytorch is installed on Linux and CPU version on Windows. To force CUDA/ROCM/CPU version to be installed, run instead:
   - CUDA (Nvidia GPUs): ``` uv pip install -e . --override override-cuda.txt --extra-index-url=https://download.pytorch.org/whl/cpu ```
   - ROCM (AMD GPUs): ``` uv pip install -e . --override override-rocm.txt --extra-index-url=https://download.pytorch.org/whl/rocm6.0 ```
   - CPU (reduced package size): ``` uv pip install -e . --override override-cpu.txt --extra-index-url=https://download.pytorch.org/whl/cu121 ```
   
   Make sure CUDA/ROCM is installed. (you can run ``` cuda-smi ``` or ``` rocm-smi```) 

# Usage

1. Make sure the virtual environment with the correct python version is activated (check installation steps 3, 4).
2. Run ``` safesight <COMMAND> ``` (for help run ``` safesight --help ```).

# Data
By default, the commands assume the dataset is in the directory `data` in the following format:
    
        data/train/accident/image1.jpg

        data/train/nonaccident/image2.jpg

        data/train/accident/...

        data/test/accident/image1.jpg

        data/test/nonaccident/image2.jpg

        data/test/accident/...

# TODO: stop calling stuff "testers", we don't have tests, we are "collecting data". test_blip -> blip (answer specific image, run on dataset, etc.), net_tester -> custom_model
