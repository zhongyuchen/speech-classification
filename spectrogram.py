from mfcc import pre_emphasis, framing, hamming_window, stft, power_spectrum
import configparser
from tqdm import tqdm
import numpy as np
import pickle
import os
from utils import get_raw_data
import librosa


def spectrogram(sample, config):
    # sample.data = sample.data[13000:19000]
    # sample.data[sample.data < 13000] = 0
    # sample.data[sample.data > 19000] = 0

    # end_point_detection(sample.data, sample.framerate, 0.03, 0.015)

    # 1
    coefficient = float(config['spectrogram']['pre_emphasis_coefficient'])
    pre_emphasis_data = pre_emphasis(sample.data, coefficient)
    # print(pre_emphasis_data.shape)

    # 2
    frame_size = float(config['spectrogram']['frame_size'])
    frame_stride = float(config['spectrogram']['frame_stride'])
    frames, frame_length = framing(pre_emphasis_data, sample.framerate, frame_size, frame_stride)
    # print('frame', frames.shape, frame_length)

    # 3
    hamming_frames = hamming_window(frames, frame_length)
    # print(hamming_frames.shape)

    # 4
    nfft = int(config['spectrogram']['nfft'])
    stft_frames = stft(hamming_frames, nfft)
    # print(stft_frames.shape)
    # print(stft_frames)

    # 5
    stft_frames /= np.max(stft_frames, axis=None)
    power_frames = power_spectrum(stft_frames, nfft)
    #print(power_frames.shape)

    return power_frames


def spectrogram_all(data, config):
    label = []
    spectrogram_feats = []
    for sample in tqdm(data, desc="spectrogram"):
        # spectrogram_feats.append(spectrogram(sample, config))
        spectrogram_feats.append(librosa.amplitude_to_db(np.abs(librosa.stft(sample.data)), ref=np.max))
        label.append(sample.label)
    return {
        'x': np.array(spectrogram_feats),
        'y': np.array(label)
    }


def dump_data(config):
    train_raw, dev_raw = get_raw_data(config)

    dev_data = spectrogram_all(dev_raw, config)
    pickle.dump(dev_data, open(os.path.join(config['spectrogram']['data_path'], config['data']['dev_file']), 'wb'))
    
    train_data = spectrogram_all(train_raw, config)
    pickle.dump(train_data, open(os.path.join(config['spectrogram']['data_path'], config['data']['train_file']), 'wb'))
    # print(train_data.shape, dev_data.shape)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    dump_data(config)
