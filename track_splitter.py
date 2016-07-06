import soundfile as sf
import numpy.fft as fft
import matplotlib.pyplot as plt

import sys
import os

if len(sys.argv) < 3:
  print 'Not enough arguments; requires filename and chunk length.'
else:
  fname = sys.argv[1]
  seconds_per_chunk = sys.argv[2]

  sig, samplerate = sf.read(fname)
  chunk_size = int(samplerate * float(seconds_per_chunk))

  position = 0
  index = 0
  if not os.path.exists(fname[:-4]):
    os.mkdir(fname[:-4])

  while position < len(sig):
    end = min(position + chunk_size, len(sig))
    sf.write(fname[:-4] + "/" + str(index) + fname[-4:],
             sig[position:end],
             samplerate)
    position = end
    index += 1
