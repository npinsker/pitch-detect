import soundfile as sf
import numpy.fft as fft
import numpy as np
import utilities

NUM_SAMPLES_PER_CHUNK = 44100 / 100

sig, samplerate = sf.read('raw/grace/62.wav')

if isinstance(sig, np.ndarray):
  sig = [(k[0] + k[1]) / 2 for k in sig]

sf.write('test_shift.wav', utilities.pitch_shift(sig, 10), samplerate)
#new_sig = utilities.pitch_shift(sig, samplerate, 2)
#sf.write('shift.wav', new_sig, samplerate)
