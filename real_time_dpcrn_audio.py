"""
#!/usr/bin/python3
#-*- coding: utf-8 -*-
@FileName: real_time_dpcrn_audio.py
@Time: 2022/11/4 14:59        
@Author:
"""
import copy
import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite
import argparse


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    '-i', '--input-device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-o', '--output-device', type=int_or_str,
    help='output device (numeric ID or substring)')

parser.add_argument('--latency', type=float, help='latency in seconds', default=0.2)
args = parser.parse_args(remaining)

# set some parameters
block_len = 512
block_shift = 256
fs_target = 16000
inp = np.zeros([1, 1, 257, 3], dtype=np.float32)
win = np.sin(np.arange(.5, block_len - .5 + 1) / block_len * np.pi)

# load models
interpreter = tflite.Interpreter(model_path='pretrained_weights/DPCRN_base/dpcrn.tflite')
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# create states for the gru
states_gru = np.zeros(input_details[1]['shape'], dtype=np.float32)
# create buffer
in_buffer = np.zeros((block_len)).astype('float32')
out_buffer = np.zeros((block_len)).astype('float32')

def callback(indata, outdata, status):
    # buffer and states to global
    global in_buffer, out_buffer, states_gru
    if status:
        print(status)
    # write to buffer
    in_buffer[:-block_shift] = in_buffer[block_shift:]
    in_buffer[-block_shift:] = np.squeeze(indata)
    # calculate fft of input block
    audio_buffer = in_buffer * win
    spec = np.fft.rfft(audio_buffer).astype('complex64')
    spec1 = copy.copy(spec)
    inp[0, 0, :, 0] = spec1.real
    inp[0, 0, :, 1] = spec1.imag
    inp[0, 0, :, 2] = 2 * np.log(abs(spec))

    # set tensors to the model
    interpreter.set_tensor(input_details[1]['index'], states_gru)
    interpreter.set_tensor(input_details[0]['index'], inp)
    # run calculation
    interpreter.invoke()
    # get the output of the model
    output_mask = interpreter.get_tensor(output_details[0]['index'])
    output_cos = interpreter.get_tensor(output_details[1]['index'])
    output_sin = interpreter.get_tensor(output_details[2]['index'])
    states_gru = interpreter.get_tensor(output_details[3]['index'])
    # calculate the ifft
    estimated_complex = spec = spec * output_mask * (output_cos + 1j*output_sin)
    estimated_block = np.fft.irfft(estimated_complex)
    out_block = estimated_block * win
    # write to buffer
    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = np.zeros((block_shift))
    out_buffer += np.squeeze(out_block)
    # output to soundcard
    outdata[:] = np.expand_dims(out_buffer[:block_shift], axis=-1)


try:
    with sd.Stream(device=(args.input_device, args.output_device),
                   samplerate=fs_target, blocksize=block_shift,
                   dtype=np.float32, latency=args.latency,
                   channels=1, callback=callback):
        print('#' * 80)
        print('press Return to quit')
        print('#' * 80)
        input()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))