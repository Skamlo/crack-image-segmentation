import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from modules.transformer import CrackImageTransform, CrackMaskTransform


class CrackTestDataset(Dataset):
    def __init__(self, dataset_path:str="./data/cracked_segmentation_dataset", load_during_init=True):
        super().__init__()
        
        # Paths
        self.__dataset_path = dataset_path
        self.__images_path = f"{self.__dataset_path}/test/images"
        self.__masks_path = f"{self.__dataset_path}/test/masks"

        # Transform images
        self.__image_transform = CrackImageTransform()
        self.__mask_transform = CrackMaskTransform()

        # Dataset
        self.__dataset_loaded = False
        self.images = []
        self.masks = []

        if load_during_init:
            self.__load_dataset()

    def __load_dataset(self):
        # Get file names
        file_names = os.listdir(self.__images_path)

        # Set pbar
        pbar = tqdm(total=len(file_names), desc="Loading test dataset", unit="files")

        # Load images
        for file_name in file_names:
            pbar.update()

            image = Image.open(f"{self.__images_path}/{file_name}").convert("RGB")
            image = self.__image_transform(image)
            self.images.append(image)

            mask = Image.open(f"{self.__masks_path}/{file_name}")
            mask = self.__mask_transform(mask)
            self.masks.append(mask)

        # Shuffle dataset
        indices = np.random.permutation(len(self.images))
        self.images = [self.images[i] for i in indices]
        self.masks = [self.masks[i] for i in indices]

        # Switch dataset as loaded
        self.__dataset_loaded = True

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if not self.__dataset_loaded:
            self.__load_dataset()

        image, mask = self.images[idx], self.masks[idx]
        return image, mask
