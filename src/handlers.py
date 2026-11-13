"""Dataset specific logic to set up each downloaded skin lesion dataset under one interface."""
import glob
import os
import shutil
from zipfile import ZipFile

import pandas as pd

from src.conf import DATA_PATH, SETUP_DATA_PATH
from src.datasets_urls import DATASETS_INFO
from src.downloaders import download_dataset


class DatasetHandler:
    """Base handler: raw downloads live in DATA_PATH/<name>, the set up copy in SETUP_DATA_PATH/<name>."""

    def __init__(self, name: str):
        self.name = name
        self.urls = DATASETS_INFO.get(self.name, {}).get("urls", ())  # dermofit and up are placed manually
        self.init_data_path = os.path.join(DATA_PATH, self.name)
        self.setup_data_path = os.path.join(SETUP_DATA_PATH, self.name)

    def download_dataset(self, force: bool = False) -> None:
        """Download the raw files of the dataset to self.init_data_path."""
        download_dataset(self.name, force=force)

    def fetch_metadata(self) -> pd.DataFrame | None:
        """Placeholder for the logic that retrieves the metadata of a dataset."""
        return None

    def setup_dataset(self, force: bool = False) -> None:
        """
        Copy the raw data to self.setup_data_path and extract every zip file in place, recursively.
        :param force: remove an existing set up copy first
        """
        if force:
            print(f"Removing old data from {self.setup_data_path}...")
            shutil.rmtree(path=self.setup_data_path, ignore_errors=True)
        print(f"Copying data from {self.init_data_path} to {self.setup_data_path}")
        shutil.copytree(src=self.init_data_path, dst=self.setup_data_path)

        def _zip_extract(folder: str) -> None:
            print(f"Extracting {folder}...")
            for file in glob.glob(os.path.join(folder, "**/*.zip"), recursive=True):
                with ZipFile(file, "r") as zip_obj:
                    zip_obj.extractall(path=os.path.split(file)[0])
                os.remove(file)
            if glob.glob(os.path.join(folder, "**/*.zip"), recursive=True):
                _zip_extract(folder=folder)

        _zip_extract(self.setup_data_path)

    def split_data(self):
        """Placeholder for the dataset specific train, validation and test split."""
        return None


class Isic16Handler(DatasetHandler):
    """ISIC 2016 challenge, part 3B (lesion classification)."""

    def __init__(self):
        super().__init__(name="isic16")
        self.image_dir = self.setup_data_path
        self.meta_dir = self.setup_data_path


class Isic17Handler(DatasetHandler):
    """ISIC 2017 challenge."""

    def __init__(self):
        super().__init__(name="isic17")
        self.image_dir = self.setup_data_path
        self.meta_dir = self.setup_data_path

    def fetch_metadata(self) -> pd.DataFrame:
        """
        Read 'ISIC-2017_Test_v2_Part3_GroundTruth.csv' and 'ISIC-2017_Test_v2_Data_metadata.csv'
        and merge them on the image id.

        :return: A dataframe with the ground truth and metadata columns and an "image_id" column
        """
        df1 = pd.read_csv(os.path.join(self.meta_dir, "ISIC-2017_Test_v2_Part3_GroundTruth.csv"),
                          index_col=0)
        df2 = pd.read_csv(os.path.join(self.image_dir, "ISIC-2017_Test_v2_Data",
                                       "ISIC-2017_Test_v2_Data_metadata.csv"), index_col=0)
        df = pd.concat([df1, df2], axis=1)
        df["image_id"] = df.index
        return df.reset_index(drop=True)


class Isic18Handler(DatasetHandler):
    """ISIC 2018 challenge."""

    def __init__(self):
        super().__init__(name="isic18")
        self.image_dir = self.setup_data_path
        self.meta_dir = self.setup_data_path


class Isic19Handler(DatasetHandler):
    """ISIC 2019 challenge."""

    def __init__(self):
        super().__init__(name="isic19")
        self.image_dir = self.setup_data_path
        self.meta_dir = self.setup_data_path

    def fetch_metadata(self) -> pd.DataFrame:
        """
        Read 'ISIC_2019_Training_GroundTruth.csv' and 'ISIC_2019_Training_Metadata.csv'
        and merge them on the image column.

        :return: A dataframe with the ground truth and metadata columns and an "image" column
        """
        df1 = pd.read_csv(os.path.join(self.meta_dir, "ISIC_2019_Training_GroundTruth.csv"),
                          index_col=0)
        df2 = pd.read_csv(os.path.join(self.meta_dir, "ISIC_2019_Training_Metadata.csv"),
                          index_col=0)
        df = pd.concat([df1, df2], axis=1)
        df["image"] = df.index
        return df.reset_index(drop=True)


class Isic20Handler(DatasetHandler):
    """ISIC 2020 challenge (JPEG images)."""

    def __init__(self):
        super().__init__(name="isic20")
        self.image_dir = os.path.join(self.setup_data_path, "train")
        self.meta_dir = os.path.join(self.setup_data_path, "ISIC_2020_Training_GroundTruth_v2.csv")

    def _remove_duplicates(self) -> pd.DataFrame:
        meta = pd.read_csv(self.meta_dir)
        duplicates = pd.read_csv(os.path.join(self.setup_data_path,
                                              "ISIC_2020_Training_Duplicates.csv"))
        return meta[~meta.loc[:, "image_name"].isin(duplicates["image_name_2"])]

    def fetch_metadata(self, remove_duplicates: bool = True) -> pd.DataFrame:
        """Read the training ground truth, without the listed duplicates by default."""
        return self._remove_duplicates() if remove_duplicates else pd.read_csv(self.meta_dir)


class SPCHandler(DatasetHandler):
    """7-point criteria evaluation database (credentials in the SPC_CREDENTIALS variable)."""

    def __init__(self):
        super().__init__(name="spc")
        self.image_folder = os.path.join(self.setup_data_path, "release_v0", "images")
        self.meta_folder = os.path.join(self.setup_data_path, "release_v0", "meta")
        self.meta_data_path = os.path.join(self.meta_folder, "meta.csv")

    def fetch_metadata(self) -> pd.DataFrame:
        """Read meta.csv of the release."""
        return pd.read_csv(self.meta_data_path)

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the metadata into train, valid and test dataframes with the
        train_indexes.csv, valid_indexes.csv and test_indexes.csv files of the release.

        :return: A tuple of train, valid and test dataframes
        """
        all_data = self.fetch_metadata()
        train_idx = pd.read_csv(os.path.join(self.meta_folder, "train_indexes.csv"))
        valid_idx = pd.read_csv(os.path.join(self.meta_folder, "valid_indexes.csv"))
        test_idx = pd.read_csv(os.path.join(self.meta_folder, "test_indexes.csv"))

        train = all_data[all_data.index.isin(train_idx["indexes"])]
        valid = all_data[all_data.index.isin(valid_idx["indexes"])]
        test = all_data[all_data.index.isin(test_idx["indexes"])]
        return train, valid, test


class DermofitHandler(DatasetHandler):
    """Dermofit image library (not downloadable, the raw data is placed manually)."""

    def __init__(self):
        super().__init__(name="dermofit")
        self.image_dir = self.setup_data_path
        self.meta_data_path = os.path.join(self.setup_data_path, "lesionlist.txt")

    def fetch_metadata(self) -> pd.DataFrame:
        """Read lesionlist.txt (image and lesion type)."""
        df = pd.read_csv(self.meta_data_path, index_col=0, header=None, sep="   ", engine="python")
        df.columns = ["image", "lesion"]
        return df


class MedNodeHandler(DatasetHandler):
    """MED-NODE dermatology database."""

    def __init__(self):
        super().__init__("mednode")
        self.image_dir = os.path.join(self.setup_data_path, "complete_mednode_dataset")
        self.meta_dir = None

    def fetch_metadata(self) -> pd.DataFrame:
        """
        Build the labels from the folder names: one row per image, with True in the "melanoma"
        or "naevus" column and 0 in the other.

        :return: A pandas dataframe with the columns "image", "melanoma" and "naevus"
        """
        melanoma_images = os.listdir(os.path.join(self.image_dir, "melanoma"))
        naevus_images = os.listdir(os.path.join(self.image_dir, "naevus"))
        melanoma_df = pd.DataFrame(melanoma_images, columns=["image"])
        melanoma_df["melanoma"] = True
        naevus_df = pd.DataFrame(naevus_images, columns=["image"])
        naevus_df["naevus"] = True
        return pd.concat([melanoma_df, naevus_df], axis=0, ignore_index=True).fillna(0)


class PadufesHandler(DatasetHandler):
    """PAD-UFES-20 smartphone images."""

    def __init__(self):
        super().__init__("padufes")
        self.image_dir = os.path.join(self.setup_data_path, "images")
        self.meta_dir = os.path.join(self.setup_data_path, "metadata.csv")

    def setup_dataset(self, force: bool = False) -> None:
        """Extract the dataset, then gather the images of the imgs_part_* folders in one folder."""
        super().setup_dataset(force=force)
        os.makedirs(self.image_dir, exist_ok=True)
        for src in glob.glob(os.path.join(self.setup_data_path, "**", "imgs_part_*", "*.png"),
                             recursive=True):
            os.rename(src=src, dst=os.path.join(self.image_dir, os.path.basename(src)))


class Ph2Handler(DatasetHandler):
    """PH2 dermoscopic image database."""

    def __init__(self):
        super().__init__("ph2")
        self.dataset_dir = os.path.join(self.setup_data_path, "PH2Dataset")
        self.image_dir = os.path.join(self.dataset_dir, "PH2 Dataset images")
        self.meta_txt = os.path.join(self.dataset_dir, "PH2_dataset.txt")
        self.meta_dir = os.path.join(self.setup_data_path, "PH2_Dataset.csv")

    def _txt_to_csv(self) -> None:
        """Write the csv equivalent of PH2_dataset.txt (the first 200 images and the header)."""
        with open(self.meta_txt, "r", encoding="utf-8") as f:
            lines = []
            for i, line in enumerate(f.readlines()):
                lines.append(line.replace("||", ",").replace("|", ",").replace(" ", "")
                             .replace(",", "", 1).replace(",\n", "\n"))
                if i > 200:
                    break
        with open(self.meta_dir, "w", encoding="utf-8") as meta_csv:
            meta_csv.writelines(lines)

    @staticmethod
    def _meta_legends() -> dict:
        rest_legend = {"A": "Absent",
                       "AT": "Atypical",
                       "P": "Present",
                       "T": "Typical"}
        other_meta_features = ["PigmentNetwork", "Dots/Globules", "Streaks", "RegressionAreas",
                               "Blue-WhitishVeil"]
        legends = {feature: rest_legend for feature in other_meta_features}
        legends["ClinicalDiagnosis"] = {0: "Common Nevus",
                                        1: "Atypical Nevus",
                                        2: "Melanoma"}
        legends["Asymmetry"] = {0: "Fully Symmetric",
                                1: "Symmetric in 1 axe",
                                2: "Fully Asymmetric"}
        legends["Colors"] = {1: "White",
                             2: "Red",
                             3: "Light - Brown",
                             4: "Dark - Brown",
                             5: "Blue - Gray",
                             6: "Black"}
        return legends

    def fetch_metadata(self) -> pd.DataFrame:
        """Read PH2_Dataset.csv, creating it from PH2_dataset.txt on the first call."""
        if not os.path.exists(self.meta_dir):
            self._txt_to_csv()
        return pd.read_csv(self.meta_dir)


class UPHandler(DatasetHandler):  # TODO Setup UP dataset
    """University of Patras dataset (not public)."""

    def __init__(self):
        super().__init__("up")


if __name__ == '__main__':
    for handler in (Isic19Handler(), Isic20Handler()):
        handler.download_dataset()
        handler.setup_dataset()
