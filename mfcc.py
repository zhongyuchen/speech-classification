from utils import get_raw_data, get_sample
import configparser
import librosa
import librosa.display
from matplotlib import pyplot as plt
import os
import numpy as np
from scipy.fftpack import dct
from tqdm import tqdm
import pickle
import torch


def pre_emphasis(data, coefficient):
    data[1:] -= coefficient * data[:-1]
    return data


def framing(data, sample_rate, frame_size, frame_stride):
    # frame size: size of a frame (s)
    # frame stride: the width of a step (s)

    # frame size and stride in time -> frame length and step in data list
    frame_length = int(round(frame_size * sample_rate))  # 480 frame size
    frame_step = int(round(frame_stride * sample_rate))  # move 240 each time
    # print(frame_length, frame_step)
    data_length = len(data)

    # num of frames (at least one frame)
    num_frames = int(np.ceil(float(data_length - frame_length) / frame_step))

    # pad data to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal
    pad_data_length = num_frames * frame_step + frame_length
    zero = np.zeros((pad_data_length - data_length))
    pad_signal = np.append(data, zero)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames, frame_length


def hamming_window(frames, frame_length):
    n = np.arange(0, frame_length)
    frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))
    return frames


def _fft(x):
    length = len(x)
    if length <= 1:
        return x
    even = _fft(x[0::2])
    odd = _fft(x[1::2])
    factor = np.exp(np.arange(0, length / 2, 1) * (-2j * np.pi / length))
    return np.hstack([even + factor * odd, even - factor * odd])


def fft(x, nfft):
    pad_x = np.hstack([x, np.zeros(nfft - len(x))])
    return _fft(pad_x)


def stft(frames, nfft):
    # fft_standard = np.fft.fft(frames[0], nfft)
    # plt.subplot(121)
    # plt.plot(fft_standard)
    # fft_my = fft(frames[0], nfft)
    # plt.subplot(122)
    # plt.plot(fft_my)
    # plt.show()

    assert np.log2(nfft) % 1 == 0
    stft_frames = []
    for frame in frames:
        stft_frames.append(fft(frame, nfft))
    return np.absolute(stft_frames)


def power_spectrum(frames, nfft):
    return (1.0 / nfft) * (frames ** 2)


def filter_banks(frames, sample_rate, nfilt, nfft):
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    # Hz -> mel: linear in mel space
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)

    # mel -> Hz
    hz_points = 700 * (np.power(10, mel_points / 2595) - 1)
    bin = np.floor((nfft + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, nfft))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    # for fb in fbank:
    #     plt.plot(fb[:240])
    # plt.xlabel('frequency')
    # plt.ylabel('amplitude')
    # plt.title('Mel-scale Filter Banks')
    # plt.show()

    filter_banks = np.dot(frames, fbank.T)
    # 0 -> a very small number
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    # filter_banks = 20 * np.log10(filter_banks)
    return filter_banks


def log(frames):
    return np.log(frames)


# def dct(frames):
#     # type II
#     mfcc = dct(frames, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]
#     return

def lifter(mfcc_feat, cep_lifter):
    (nframes, ncoeff) = mfcc_feat.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc_feat *= lift
    return mfcc_feat


def short_time_energy(frames, data, framerate, frame_size, frame_stride, energy_threshold):
    frame_length = frames.shape[1]
    def frame2data(idx):
        return int(idx * frame_length / 2)

    energy = np.sum(np.square(frames), axis=1)
    energy /= np.max(energy)
    to_index = np.arange(len(energy))
    # print(energy>energy_threshold)
    idx = to_index[energy>energy_threshold]
    # print(idx[0], idx[-1])
    # print(energy.shape)
    # plt.subplot(211)
    # plt.plot(np.arange(0, idx[0]), energy[0:idx[0]], 'b')
    # plt.plot(np.arange(idx[0]-1, idx[-1]+1), energy[idx[0]-1:idx[-1]+1], 'r')
    # plt.plot(np.arange(idx[-1], len(energy)), energy[idx[-1]:], 'b')
    # plt.title('Short Time Energy')
    # plt.subplot(212)
    #
    start_idx = frame2data(idx[0])
    end_idx = frame2data(idx[-1])
    # print(start_idx, end_idx)
    # plt.plot(np.arange(0, start_idx), data[0:start_idx], 'b')
    # plt.plot(np.arange(start_idx-1, end_idx+1), data[start_idx-1:end_idx+1], 'r')
    # plt.plot(np.arange(end_idx, len(data)), data[end_idx:], 'b')
    # plt.title('Data')
    # plt.show()
    # exit()
    return [idx[0], idx[-1]], [start_idx, end_idx]


def zero_crossing_rate(frames, data, frame_idx, data_idx, zrc_threshold):
    signs = np.sign(frames)
    zrc = np.sum(np.abs(signs[:, 1:] - signs[:, :-1]), axis=1) / 2
    frame_length = frames.shape[1]

    def frame2data(idx):
        return int(idx * frame_length / 2)

    # to_index = np.arange(len(zrc))
    # idx= to_index[zrc<zrc_threshold]

    start_idx = frame_idx[0]
    while zrc[start_idx] < zrc_threshold and start_idx > 0:
        start_idx -= 1
    end_idx = frame_idx[1]
    while zrc[end_idx] < zrc_threshold and end_idx < len(zrc):
        end_idx += 1

    # plt.subplot(211)
    # plt.plot(np.arange(0, start_idx + 1), zrc[:start_idx + 1], 'b')
    # plt.plot(np.arange(start_idx, end_idx), zrc[start_idx:end_idx], 'r')
    # plt.plot(np.arange(end_idx - 1, len(zrc)), zrc[end_idx - 1:], 'b')
    # plt.title('End Point Detection')
    # plt.subplot(212)

    start_idx = frame2data(start_idx)
    end_idx = frame2data(end_idx)
    # print(start_idx, end_idx)
    # plt.plot(np.arange(0, start_idx + 1), data[0:start_idx + 1], 'b')
    # plt.plot(np.arange(start_idx, end_idx), data[start_idx:end_idx], 'r')
    # plt.plot(np.arange(end_idx-1, len(data)), data[end_idx-1:], 'b')
    # plt.title('Data')
    # plt.plot(data)
    # plt.show()
    # exit()
    return [start_idx, end_idx]


def end_point_detection(data, framerate, frame_size, frame_stride):
    energy_threshold = 0.1
    zrc_threshold = 70
    frames, frame_length = framing(data, framerate, frame_size, frame_stride)
    frames = hamming_window(frames, frame_length)
    print(frames.shape)

    time_idx, data_idx = short_time_energy(frames, data, framerate, frame_size, frame_stride, energy_threshold)
    [start_idx, end_idx] = zero_crossing_rate(frames, data, time_idx, data_idx, zrc_threshold)

    # plt.plot(np.arange(0, start_idx), data[0:start_idx], 'b')
    # plt.plot(np.arange(start_idx -1, end_idx+1), data[start_idx-1:end_idx+1], 'r')
    # plt.plot(np.arange(end_idx, len(data)), np.zeros(len(data)-end_idx), 'b')
    # plt.title('Data')
    # plt.plot(data)
    # plt.show()
    # exit()


def mfcc(sample, config):
    # sample.data = sample.data[13000:19000]
    sample.data[sample.data < 11000] = 0
    sample.data[sample.data > 20000] = 0

    # end_point_detection(sample.data, sample.framerate, 0.03, 0.015)

    # 1
    coefficient = float(config['mfcc']['pre_emphasis_coefficient'])
    # plt.plot(sample.data, label='original')
    pre_emphasis_data = pre_emphasis(sample.data, coefficient)
    # plt.plot(pre_emphasis_data, label='pre emphasis')
    # plt.title('Pre Emphasis')
    # plt.legend()
    # plt.show()
    # exit()
    # print(pre_emphasis_data.shape)

    # 2
    frame_size = float(config['mfcc']['frame_size'])
    frame_stride = float(config['mfcc']['frame_stride'])
    frames, frame_length = framing(pre_emphasis_data, sample.framerate, frame_size, frame_stride)
    print('frame', frames.shape, frame_length)

    # 3
    hamming_frames = hamming_window(frames, frame_length)
    # print(hamming_frames.shape)

    # 4
    nfft = int(config['mfcc']['nfft'])
    stft_frames = stft(hamming_frames, nfft)
    # print(stft_frames.shape)
    # print(stft_frames)
    # stft_frames -= (np.mean(stft_frames, axis=0) + 1e-8)
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(stft_frames, y_axis='linear')
    # plt.colorbar()
    # plt.xlabel('Frame')
    # plt.title('STFT')
    # plt.show()
    # exit()

    # 5
    power_frames = power_spectrum(stft_frames, nfft)
    # print(power_frames.shape)
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(power_frames, y_axis='linear')
    # plt.colorbar()
    # plt.xlabel('Frame')
    # plt.title('Power Spectrum')
    # plt.show()
    # exit()

    # 6
    nfilt = int(config['mfcc']['nfilt'])
    filter_frames = filter_banks(power_frames, sample.framerate, nfilt, nfft)
    # print(filter_frames.shape)
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(filter_frames, y_axis='linear')
    # plt.colorbar()
    # plt.xlabel('Frame')
    # plt.title('Power Spectrum')
    # plt.show()
    # exit()

    # plt.figure(figsize=(10, 4))
    # S = librosa.feature.melspectrogram(y=sample.data, sr=sample.framerate, n_mels=40,fmax = 16000)
    # librosa.display.specshow(librosa.power_to_db(S,ref = np.max),y_axis = 'mel', fmax = 8000,x_axis = 'time')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel spectrogram')
    # plt.tight_layout()
    # plt.show()
    # exit()

    # 7
    log_frames = log(filter_frames)
    print(log_frames.shape)

    # D = librosa.amplitude_to_db(np.abs(librosa.stft(sample.data)), ref=np.max)
    # print(D.shape)
    # D = D[:,20:40]
    # librosa.display.specshow(D, y_axis='log')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Log-frequency power spectrogram')
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(log_frames, x_axis='time')
    # plt.colorbar()
    # plt.xlabel('Frame')
    # plt.title('Log Power Spectrum')
    # plt.specgram(sample.data, Fs=sample.framerate, scale_by_freq=True, sides='default')
    # plt.show()
    # exit()

    # 8
    nceps = int(config['mfcc']['nceps'])
    mfcc_feat = dct(frames, type=2, axis=1, norm='ortho')[:, 1: (nceps + 1)]
    print(mfcc_feat.shape)

    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(mfcc_feat, x_axis='time')
    # plt.colorbar()
    # plt.xlabel('Frame')
    # plt.title('MFCC')
    # plt.tight_layout()
    # plt.show()
    # exit()

    # 9
    cep_lifter = int(config['mfcc']['cep_lifter'])
    mfcc_feat = lifter(mfcc_feat, cep_lifter)
    # mfcc_feat -= (np.mean(mfcc_feat, axis=0) + 1e-8)
    # mfcc_feat = librosa.feature.delta(mfcc_feat)
    print(mfcc_feat.shape)

    # mfcc_feat = librosa.feature.mfcc(y=sample.data, sr=sample.framerate, n_mfcc=12)
    # plt.plot(sample.data)
    # plt.show()
    # mfcc_feat = mfcc_feat.T
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(mfcc_feat.T, x_axis='time')
    # plt.colorbar()
    # plt.xlabel('Frame')
    # plt.title('MFCC')
    # plt.tight_layout()
    # plt.show()
    # exit()
    return mfcc_feat


def mfcc_one(folder_path, file, time_length=2.0):
    sample = get_sample(folder_path, file, time_length, test=True)
    feat = librosa.feature.mfcc(y=sample.data, sr=sample.framerate, n_mfcc=12)
    feat = np.array([feat])
    feat = torch.from_numpy(feat).type('torch.FloatTensor')
    return torch.tensor(feat)


def mfcc_all(data, config):
    label = []
    mfcc_feats = []
    for sample in tqdm(data, desc="mfcc"):
        # mfcc_feats.append(mfcc(sample, config))
        mfcc_feats.append(librosa.feature.mfcc(y=sample.data, sr=sample.framerate, n_mfcc=12))
        label.append(sample.label)
    return {
        'x': np.array(mfcc_feats),
        'y': np.array(label)
    }


def dump_data(config):
    train_raw, dev_raw = get_raw_data(config)

    dev_data = mfcc_all(dev_raw, config)
    pickle.dump(dev_data, open(os.path.join(config['mfcc']['data_path'], config['data']['dev_file']), 'wb'))

    train_data = mfcc_all(train_raw, config)
    pickle.dump(train_data, open(os.path.join(config['mfcc']['data_path'], config['data']['train_file']), 'wb'))


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    dump_data(config)
    exit()

    sample = get_sample(os.path.join(config['data']['raw_data_path'], config['data']['sample'][:11]), config['data']['sample'], float(config['data']['time_length']))
    # mfcc(sample, config)
    # exit()
    # print(sample.framerate)
    # print(sample.data.shape)
    # print(sample.id)
    # print(sample.label)
    # print(sample.time)
    # print(sample.nframes)

    # y, sr = librosa.load(sample.data, offset=30, duration=5)
    mfcc = librosa.feature.mfcc(y=sample.data, sr=sample.framerate, n_mfcc=12)
    print(mfcc.shape)
    exit()
    # y, sr = librosa.load(librosa.util.example_audio_file(), offset=30, duration=5)
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
    print(mfcc.shape)  # 20x216
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()
    exit()

    # print(len(sample.data))
    # plt.plot(sample.data)
    # plt.show()
    # zcr = librosa.feature.zero_crossing_rate(sample.data, frame_length=256, hop_length=256)
    # plt.subplot(211)
    # plt.plot(sample.time, sample.data)
    # plt.title(config['data']['sample'])
    # plt.subplot(212)
    # print(zcr.shape)
    # plt.plot(zcr[0])
    # plt.show()
