import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from modules.transformer import CrackImageTransform


class CrackTrainDataset(Dataset):
    def __init__(self, dataset_path:str="data/crack_segmentation_dataset"):
        super().__init__()

        # Paths
        self.__dataset_path = dataset_path
        self.__images_path = f"{self.__dataset_path}/train/images"

        # Group split
        self.__cracked_groups = {
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
        }
        self.__non_cracked_groups = {
            "noncracknoncrackconcretewall"
        }

        # Transform images
        self.__transform = CrackImageTransform()

        # Data container: List of tuples [(path_to_image, label_tensor), ...]
        self.samples = []
        
        # Prepare file list immediately
        self.__prepare_dataset()

    def __get_group_name(self, file_name:str):
        return "".join([ch.rstrip(".jpg") for ch in file_name if not ch.isdigit() and ch not in "_-"])
    
    def __prepare_dataset(self):
        if not os.path.exists(self.__images_path):
            raise FileNotFoundError(f"Path not found: {self.__images_path}")
            
        file_names = os.listdir(self.__images_path)

        cracked_files = []
        non_cracked_files = []

        # Sort files into groups
        print("Indexing dataset files...")
        for file_name in file_names:
            group_name = self.__get_group_name(file_name)
            if group_name in self.__cracked_groups:
                cracked_files.append(file_name)
            elif group_name in self.__non_cracked_groups:
                non_cracked_files.append(file_name)

        # Add Cracked
        for file_name in cracked_files:
            full_path = os.path.join(self.__images_path, file_name)
            label = torch.tensor([1.0], dtype=torch.float32)
            self.samples.append((full_path, label))

        # Add Non-Cracked
        for file_name in non_cracked_files:
            full_path = os.path.join(self.__images_path, file_name)
            label = torch.tensor([0.0], dtype=torch.float32)
            self.samples.append((full_path, label))

        # Shuffle the combined list
        np.random.shuffle(self.samples)
        
        print(f"Dataset ready. Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        image = self.__transform(image)

        return image, label
