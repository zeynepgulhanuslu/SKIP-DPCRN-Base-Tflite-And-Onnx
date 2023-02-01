import copy
import os
import time

import numpy as np
import onnxruntime
import soundfile as sf
import yaml

from DPCRN_base import DPCRN_model
from real_time_onnx import mk_mask_mag, mk_mask_pha


def enhance_with_onnx(model, noisy_file, out_audio_file):
    start = time.time()
    # load model
    interpreter_1 = onnxruntime.InferenceSession(model)
    model_input_names = [inp.name for inp in interpreter_1.get_inputs()]
    # preallocate input
    model_inputs = {
        inp.name: np.zeros(
            [dim if isinstance(dim, int) else 1 for dim in inp.shape],
            dtype=np.float32)
        for inp in interpreter_1.get_inputs()}

    audio, fs = sf.read(noisy_file)
    # create states for the lstms
    audio_length = audio.shape[0]
    print(f'audio length: {audio_length}')
    inp_len = int(audio_length / 2) + 1
    inp = np.zeros([1, 1, inp_len, 3], dtype=np.float32)
    # load audio file at 16k fs (please change)
    win = np.sin(np.arange(.5, audio_length - .5 + 1) / audio_length * np.pi)
    # check for sampling rate
    if fs != 16000:
        raise ValueError('This model only supports 16k sampling rate.')
    # preallocate output audio
    audio_buffer = audio * win
    print(f'audio buffer len:{audio_buffer}')
    spec = np.fft.rfft(audio_buffer).astype('complex64')
    print(f'spec size:{len(spec)}')
    spec1 = copy.copy(spec)
    inp[0, 0, :, 0] = spec1.real
    inp[0, 0, :, 1] = spec1.imag
    inp[0, 0, :, 2] = 2 * np.log(abs(spec))
    print(len(model_inputs))
    # set block to input
    model_inputs[model_input_names[0]] = inp
    # run calculation
    model_outputs = interpreter_1.run(None, model_inputs)
    model_inputs[model_input_names[1]] = model_outputs[3]
    output_mask = model_outputs[0]
    output_cos = model_outputs[1]
    output_sin = model_outputs[2]
    # calculate the ifft
    estimated_real, estimated_imag = mk_mask_mag([spec.real, spec.imag, output_mask])
    enh_real, enh_imag = mk_mask_pha([estimated_real, estimated_imag, output_cos, output_sin])
    estimated_complex = enh_real + 1j * enh_imag
    estimated_block = np.fft.irfft(estimated_complex)
    estimated_block = estimated_block * win
    # write to .wav file
    sf.write(out_audio_file, estimated_block, fs)
    print('Processing Time [ms]:')
    print(time.time() - start)
    print('Processing finished.')


def enhance_with_original_model(config_file, model_file, noisy_file, out_file, plot):
    f = open(config_file, 'r', encoding='utf-8')
    result = f.read()
    config_dict = yaml.safe_load(result)

    dpcrn_model = DPCRN_model(batch_size=1, length_in_s=10, lr=1e-3, config=config_dict)
    dpcrn_model.build_DPCRN_model()
    dpcrn_model.model.load_weights(model_file)
    dpcrn_model.enhancement(noisy_f=os.path.join(noisy_file), output_f=os.path.join(out_file), plot=plot)


if __name__ == '__main__':
    model_file = 'D:/zeynep/data/noise-cancelling/dpcrn-data/sau-250k-v1/sau-250k-v1model_08_0.000049.h5'
    noisy_file = 'D:/zeynep/data/noise-cancelling/audio-samples/real-recordings/noisy/audioset_realrec_airconditioner_2TE3LoA2OUQ_noisy.wav'
    out_file = 'D:/zeynep/data/noise-cancelling/audio-samples/real-recordings/dpcrn/sau-250k-v1/epoch-8/audioset_realrec_airconditioner_2TE3LoA2OUQ_noisy.wav'
    #enhance_with_onnx(model_file, noisy_file, out_file)

    enhance_with_original_model('configuration/DPCRN-custom.yaml', model_file, noisy_file, out_file, True)
