import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from modules.transformer import CrackImageTransform


class CrackTrainDataset(Dataset):
    def __init__(self, dataset_path:str="./data/cracked_segmentation_dataset", load_during_init=True):
        super().__init__()

        # Paths
        self.__dataset_path = dataset_path
        self.__images_path = f"{self.__dataset_path}/train/images"

        # Group split
        self.__cracked_groups = [
            "CFD",
            "CRACK",
            "CRACKIMG",
            "DeeCrack",
            "DeeCrackIMG",
            "DeeCrackQA",
            "EuenMuller",
            "EuenMullera",
            "GAPStest",
            "GAPStrain",
            "GAPSvalid",
            "RissbilderforFlorianSA",
            "SylvieChambon",
            "SylvieChambonImGTESARa",
            "SylvieChambonImnoGTESARa",
            "VolkerDSC",
            "cracktree",
            "forest"
        ]
        self.__non_cracked_groups = [
            "noncracknoncrackconcretewall"
        ]

        # Transform images
        self.__transform = CrackImageTransform()

        # Dataset
        self.__dataset_loaded = False
        self.images = []
        self.labels = []

        if load_during_init:
            self.__load_dataset()

    def __get_group_name(self, file_name:str):
        return "".join([ch.rstrip(".jpg") for ch in file_name if not ch.isdigit() and not ch in "_-"])
    
    def __load_dataset(self):
        # Get file names and split by whether they cracked
        file_names = os.listdir(self.__images_path)

        crakced_files_names = []
        non_crakced_files_names = []

        for file_name in file_names:
            group_name = self.__get_group_name(file_name)
            if group_name in self.__cracked_groups:
                crakced_files_names.append(file_name)
            elif group_name in self.__non_cracked_groups:
                non_crakced_files_names.append(file_name)

        # Set pbar
        n_cracked = len(crakced_files_names)
        n_non_cracked = len(non_crakced_files_names)
        pbar = tqdm(total=n_cracked + n_non_cracked, desc="Loading train dataset", unit="files")

        # Load cracked images
        for file_name in crakced_files_names:
            pbar.update()

            image = Image.open(f"{self.__images_path}/{file_name}").convert("RGB")
            image = self.__transform(image)

            self.images.append(image)
            self.labels.append(torch.tensor([1.0], dtype=torch.float32))

        # Load con-cracked images
        for file_name in non_crakced_files_names:
            pbar.update()

            image = Image.open(f"{self.__images_path}/{file_name}").convert("RGB")
            image = self.__transform(image)

            self.images.append(image)
            self.labels.append(torch.tensor([0.0], dtype=torch.float32))            

        # Shuffle dataset
        indices = np.random.permutation(len(self.images))
        self.images = [self.images[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

        # Switch dataset as loaded
        self.__dataset_loaded = True

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if not self.__dataset_loaded:
            self.__load_dataset()

        image, label = self.images[idx], self.labels[idx]
        return image, label
