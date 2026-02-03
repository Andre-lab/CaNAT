import torch
from torch.utils.data import Dataset
import pandas as pd


class AARNADataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, label_col_name: str, transform=None, common_length: int = 100):
        """
        Initialize the AARNADataset.

        Args:
            data_frame (pd.DataFrame): DataFrame containing the label data.
            label_col_name (str): Name of the column with label data.
            transform (callable, optional): A function to transform the label data 
                                            and calculate input data.
            common_length (int, optional): Fixed length for sequence transformation. Default is 100.
        """
        self.data_frame = data_frame
        self.label_col_name = label_col_name
        self.transform = transform
        self.common_length = common_length

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.data_frame)

    def __getitem__(self, idx: int):
        """
        Retrieve a single sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            (torch.Tensor, torch.Tensor): Input data and label data as tensors.
        """
        # Retrieve label data from the DataFrame
        label_data = self.data_frame.iloc[idx][self.label_col_name]
        
        # Apply transformation to generate input data
        if self.transform:
            input_data, label_data = self.transform(label_data, self.common_length)
        else:
            raise ValueError("Transform function must be provided to generate input data.")

        return torch.tensor(input_data), torch.tensor(label_data)
