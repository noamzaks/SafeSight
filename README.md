# SafeSight
Detecting Suspicious Events in Video.

## Installation

1. Clone the repository ``` git clone https://github.com/noamzaks/SafeSight.git ```
2. Run the installation script[^1] ``` ./bootstrap ```
3. Activate Python virtual environment ``` source <venv directory>/bin/activate ```

[^1]: The installation script autodetects CUDA 10.1+/ROCm 6.0 installations. For advanced installation check ``` ./bootstrap -h```

## Usage

1. Make sure the virtual environment with the correct python version is activated (check installation step 3). You can always install additional versions with ``` ./bootstrap -p <version> ```
2. Run ``` safesight <COMMAND> ``` (for help run ``` safesight --help ```).

## Data
By default, the commands assume the dataset is in the directory `data` in the following format:
    
        data/train/accident/image1.jpg

        data/train/nonaccident/image2.jpg

        data/train/accident/...

        data/test/accident/image1.jpg

        data/test/nonaccident/image2.jpg

        data/test/accident/...