from modules.dataset import CrackTestDataset
from torch.utils.data import DataLoader


class CrackTestDataloader(DataLoader):
    def __init__(self, batch_size:int=1, *args, **kwargs):
        super().__init__(CrackTestDataset(*args, **kwargs), batch_size=batch_size)
