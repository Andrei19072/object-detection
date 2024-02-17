# Object Detection

### Setup

First, install python 3.11 from https://www.python.org/downloads/ and make sure you check the 'install pip' option in the installer.

Next, run the following commands:

python -m venv venv\
venv/Scripts/activate (might be different on mac)\
pip install -r requirements.txt

The run 'python main.py' to run the code.

### Dataset

The dataset will be very big so we should store it locally instead of on github. Run either 'python download.py' to download just the validation part of the dataset (to save space) or 'python download.py --all' to download the entire dataset. This will place it in the data/ folder, which will be automatically ignored by git when you commit any changes.