import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
import tarfile
import gzip
from file_processing import read_bytes
import pdb


class NimrodDataset(Dataset):
    def __init__(self, date_range, root_dir, stride=5, seq_len=25):
        self.start_date = date_range[0]
        self.end_date = date_range[1]
        self.root_dir = root_dir
        self.stride = stride
        self.seq_len = seq_len
        self.height = 512
        self.width = 512
        self.pos = (149000, 544000)

    def __len__(self):
        return int(((self.end_date - self.start_date).days - self.seq_len) * 288 / self.stride + 1)

    def __getitem__(self, idx):
        start_timestamp = self.start_date + \
            timedelta(minutes=5 * self.stride * idx)
        end_timestamp = start_timestamp + \
            timedelta(minutes=5 * (self.seq_len - 1))
        file_path = os.path.join(self.root_dir, str(
            start_timestamp.year), f"metoffice-c-band-rain-radar_uk_{start_timestamp.strftime('%Y%m%d')}_1km-composite.dat.gz.tar")
        daily_data = tarfile.open(file_path)
        data = np.full((self.seq_len, self.height, self.width), -1)
        current_date = start_timestamp.date()
        for i, timestamp in enumerate(pd.date_range(start_timestamp, end_timestamp, freq='5T')):
            if timestamp.date() != current_date:
                file_path = os.path.join(self.root_dir, str(
                    end_timestamp.year), f"metoffice-c-band-rain-radar_uk_{end_timestamp.strftime('%Y%m%d')}_1km-composite.dat.gz.tar")
                daily_data = tarfile.open(file_path)
            try:
                s = read_bytes(gzip.open(daily_data.extractfile(
                    f"metoffice-c-band-rain-radar_uk_{timestamp.strftime('%Y%m%d%H%M')}_1km-composite.dat.gz"), mode='rb').read, self.height, self.width, self.pos)
                data[i] = s
            except:
                pass
            current_date = timestamp.date()
        del daily_data
        return data

#time_range = pd.date_range(datetime(2020, 1, 1), datetime(2020, 12, 31))
#dataset = NimrodDataset(time_range, "./Nimrod")


if __name__ == '__main__':
    dataset = NimrodDataset(
        [datetime(2020, 1, 1), datetime(2020, 12, 31)], "./Nimrod_2014-2020")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    data_iter = iter(dataloader)
    for i in range(10):
        data = next(data_iter)
        print(data.shape)
        print(data[:][0])
    pdb.set_trace()
