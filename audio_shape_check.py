import soundfile as sf
if __name__ == '__main__':
    p = 'D:/zeynep/data/noise-cancelling/dpcrn-data/clean/train/common_voice_tr_19302365.wav'
    noise_f = sf.read(p, dtype='float32', start=0, stop=0 + 16000)[0]
    print(noise_f.shape)
    if noise_f.shape[0] == 16000:
        print('shape is true')