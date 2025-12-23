import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from modules.transformer import CrackImageTransform
from modules.transformer import CrackMaskTransform 


class CrackTestDataset(Dataset):
    def __init__(self, dataset_path:str="data/crack_segmentation_dataset"):
        super().__init__()
        
        # Paths
        self.__dataset_path = dataset_path
        self.__images_path = f"{self.__dataset_path}/test/images"
        self.__masks_path = f"{self.__dataset_path}/test/masks"

        # Transforms
        self.__image_transform = CrackImageTransform()
        self.__mask_transform = CrackMaskTransform()

        # Data container: List of tuples [(image_path, mask_path), ...]
        self.samples = []

        # Index files immediately
        self.__prepare_dataset()

    def __prepare_dataset(self):
        if not os.path.exists(self.__images_path) or not os.path.exists(self.__masks_path):
            raise FileNotFoundError(f"Test paths not found in {self.__dataset_path}")

        # Get file names
        file_names = os.listdir(self.__images_path)
        
        print(f"Indexing test dataset ({len(file_names)} files)...")

        # Store paths only
        for file_name in file_names:
            img_full_path = os.path.join(self.__images_path, file_name)
            mask_full_path = os.path.join(self.__masks_path, file_name)
            
            if os.path.exists(mask_full_path):
                self.samples.append((img_full_path, mask_full_path))
            else:
                print(f"Warning: Mask not found for {file_name}, skipping.")

        np.random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)
        except Exception as e:
            print(f"Error loading sample {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        image = self.__image_transform(image)
        mask = self.__mask_transform(mask)

        return image, mask
