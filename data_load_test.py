import os

import numpy as np

from data_loader import mk_mixture, add_pyreverb, fir
import soundfile as sf


def mix_audio_test(clean_f, noise_f, SNR, out_file):
    clean_s = np.load(clean_f) / 32768.0
    #clean_s = add_pyreverb(clean_s, fir)
    noise_s = sf.read(noise_f, dtype='float32')[0]
    clean_s, noise_s, noisy_s, _ = mk_mixture(clean_s, noise_s, SNR, eps=1e-8)
    sf.write(out_file, clean_s, 16000)


def load_npy_file(npy_file):
    data = np.load(npy_file)
    print(type(data), len(data))
    print(data)


if __name__ == '__main__':
    root = 'D:/zeynep/data/noise-cancelling/dpcrn-data/'
    clean_f = os.path.join(root, '1000.npy')
    noise_f = os.path.join(root, 'noise/000a65e8-1f64-4bf6-bb7c-264cba6e2722_snr15dB.wav')
    out_f = os.path.join(root, 'sample_noisy.wav')
    mix_audio_test(clean_f, noise_f, 0, out_f)
    # load_npy_file(os.path.join(root, 'sample-data/train/30.npy'))
