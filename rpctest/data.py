import torch.utils.data as data
import pandas as pd
import glob
import os

class GeneralPytorchDataset(data.Dataset):
    def __init__(self, mode, pickled_data_path):
        all_files = glob.glob(os.path.join(pickled_data_path , "*.pkl"))
        self.data_df = pd.concat((pd.read_pickle(f) for f in all_files))
        self.data_df = self.data_df.reset_index(drop=True)
        self.mode = mode
        
    def __getitem__(self, index):
        input_tensor = self.data_df[[0]][0][index]
        output_tensor = self.data_df[[1]][1][index]
        return input_tensor, output_tensor
    
    def __len__(self):
        return len(self.data_df)


def get_dataset(data_file, mode):
    dataset = GeneralPytorchDataset(mode, data_file)
    return dataset
