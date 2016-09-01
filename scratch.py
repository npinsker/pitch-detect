import soundfile as sf
import numpy.fft as fft
import numpy as np
import utilities
import data_reader

NUM_SAMPLES_PER_CHUNK = 44100 / 100

#sig, samplerate = sf.read('raw/af.wav')

#if isinstance(sig, np.ndarray):
#  sig = [(k[0] + k[1]) / 2 for k in sig]
sig, samplerate = sf.read('shift.wav')
sf.write('shift2.wav', sig, samplerate)
print sig, 'done'
block = data_reader.read_data_with_flat_pitch('raw/a0.wav', 'a4')
print block[0][1][0:10]
sf.write('test.wav', block[0][1], 44100)
#for i in range(2, 12):
#  print 'processing pitchshift by %d' % i
#  sf.write('af%d.wav' % i, utilities.pitch_shift(sig, i), samplerate)
#new_sig = utilities.pitch_shift(sig, samplerate, 2)
#sf.write('shift.wav', new_sig, samplerate)
