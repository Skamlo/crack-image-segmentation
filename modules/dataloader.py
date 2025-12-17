from modules.dataset import CrackTrainDataset, CrackTestDataset
from torch.utils.data import DataLoader


class CrackTrainDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(CrackTrainDataset(), *args, **kwargs)


class CrackTestDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(CrackTestDataset(), *args, **kwargs)
