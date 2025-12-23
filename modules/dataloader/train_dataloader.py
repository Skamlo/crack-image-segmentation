from modules.dataset import CrackTrainDataset
from torch.utils.data import DataLoader


class CrackTrainDataloader(DataLoader):
    def __init__(self, batch_size:int=16, *args, **kwargs):
        super().__init__(CrackTrainDataset(*args, **kwargs), batch_size=batch_size)
