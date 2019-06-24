import configparser
import os
import numpy as np
from tqdm import tqdm
import wave


class Sample:
    def __init__(self, nframes, framerate, data, id, label):
        self.label = label  # label
        self.id = id  # speaker
        self.data = np.array(data) * 1.0  # signal data
        self.nframes = nframes  # length of data
        self.framerate = framerate  # sample rate
        self.time = np.arange(0, nframes) * (1.0 / framerate)  # time list


def check_folder_name(folders):
    new_folders = []
    for folder in folders:
        if len(folder) == 11 and int(folder):
            new_folders.append(folder)
        else:
            print("ignore", folder)
    return new_folders


def check_sample_name(samples):
    new_samples = []
    for sample in samples:
        if len(sample) == 21 and sample[11] == sample[14] == '-' and sample[-4:] == ".wav":
            new_samples.append(sample)
        else:
            print("ignore", sample)
    return new_samples


def fetch_data(data, length):
    if len(data) < length:
        data = np.hstack([data, np.zeros(length - len(data))])
    else:
        data = data[:length]
    return data


def get_sample(folder_path, file, time_length, test=False):
    with wave.open(os.path.join(folder_path, file), "rb") as f:
        nframes = f.getnframes()
        framerate = f.getframerate()
        length = int(time_length * framerate)
        data = fetch_data(np.fromstring(f.readframes(nframes), dtype=np.short), length)
        if not test:
            id = file[:11]
            label = int(file[12:14])
        else:
            id = '16307130194'
            label = 0
        sample = Sample(nframes, framerate, data, id, label)
        return sample


def get_raw_data(config):
    np.random.seed(0)

    raw_path = config["data"]["raw_data_path"]
    folders = check_folder_name(os.listdir(raw_path))
    print("folders", len(folders))

    train_data = []
    dev_data = []
    dev_ratio = float(config["data"]["dev_ratio"])
    time_length = float(config['data']['time_length'])

    for folder in tqdm(folders, desc="getting data"):
        folder_path = os.path.join(raw_path, folder)
        files = check_sample_name(os.listdir(folder_path))
        dev_flag = np.random.rand() <= dev_ratio
        for file in files:
            sample = get_sample(folder_path, file, time_length)
            if dev_flag:
                dev_data.append(sample)
            else:
                train_data.append(sample)

    print("train size:", len(train_data), "dev size:", len(dev_data))
    return train_data, dev_data
