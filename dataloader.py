import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np




class NPZDataset(Dataset):
    def __init__(self, path:str, transform=None):
        self.path = path
        self.files = list(Path(path).glob('*.npz'))
        self.data = np.load(str(self.files[0]))['arr_1']
        self.transform = transform

    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):

        # label "optical properties"
        torch_array_input = torch.from_numpy(np.load(str(self.files[0]))['arr_1'][item]).permute(2, 0, 1)
        # optical distributions
        torch_array_output = torch.from_numpy(np.load(str(self.files[0]))['arr_0'][item]).permute(2, 0, 1)

        return (torch_array_input, torch_array_output)

