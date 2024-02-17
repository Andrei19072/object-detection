import sys
import gdown
import os
import zipfile

if not os.path.exists("data"):
    os.mkdir("data")

def download(filename, id):
    path = f"data/{filename}"
    gdown.download(id=id, output=path)
    print(f"{filename} successfully downloaded...")

def download_zip(filename, id):
    path = f"data/{filename}.zip"
    gdown.download(id=id, output=path)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall("data")
    os.remove(path)
    print(f"{filename} successfully downloaded...")

download("annotation_val.odgt", "10WIRwu8ju8GRLuCkZ_vT6hnNxs5ptwoL")
download_zip("CrowdHuman_val", "18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO")
    
if len(sys.argv) > 1 and sys.argv[1] == "--all":
    download("annotation_train.odgt", "1UUTea5mYqvlUObsC1Z8CFldHJAtLtMX3")
    download_zip("CrowdHuman_train01", "134QOvaatwKdy0iIeNqA_p-xkAhkV4F8Y")
    download_zip("CrowdHuman_train02", "17evzPh7gc1JBNvnW1ENXLy5Kr4Q_Nnla")
    download_zip("CrowdHuman_train03", "1tdp0UCgxrqy1B6p8LkR-Iy0aIJ8l4fJW")
