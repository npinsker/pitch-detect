import soundfile as sf
import numpy.fft as fft
import matplotlib.pyplot as plt

NUM_SAMPLES_PER_CHUNK = 44100 / 100

sig, samplerate = sf.read('raw/sine_a3.wav')

transformed = fft.fft(sig[:NUM_SAMPLES_PER_CHUNK])
plt.plot(transformed[0:20])
plt.show()
