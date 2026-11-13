"""Download every dataset of DATASETS_INFO to data/."""
from src.datasets_urls import DATASETS_INFO
from src.downloaders import download_dataset

for dataset in DATASETS_INFO:
    download_dataset(dataset, force=False)
