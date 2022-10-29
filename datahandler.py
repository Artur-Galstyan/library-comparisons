from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
from torch.utils.data.dataset import Dataset
import torch.utils.data
from torch.utils.data import DataLoader


class HeartData(Dataset):
    def __init__(self, file_path):
        raw_data = pd.read_csv(file_path)
        x = raw_data.values[:, :-1]
        y = raw_data.values[:, -1]
        y = y.reshape(len(y), 1)
        one_hot_encoder = OneHotEncoder()
        min_max_scaler = MinMaxScaler()
        one_hot_encoder.fit(y)
        y = one_hot_encoder.transform(y).toarray()
        self.x = min_max_scaler.fit_transform(x)
        self.y = y

    def __len__(self):
        # e.g. len(x)
        return len(self.x)

    def __getitem__(self, idx):
        # e.g. foo[12]
        # idx would be 12 then
        return self.x[idx], self.y[idx]


def get_data():
    heart_data = HeartData("heart.csv")

    training_data, test_data = torch.utils.data.random_split(
        heart_data,
        [int(len(heart_data) * 0.7), len(heart_data) - int(len(heart_data) * 0.7)],
    )
    train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)
    return train_dataloader, test_dataloader
