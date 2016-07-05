import soundfile as sf
import numpy as np
import numpy.fft as fft
import sys
from scipy import *
from pylab import *
from numpy import interp

N = 2048
H = N/4

# Given a signal, creates a signal that is `tscale` times shorter
# without changing the pitch.
# taken from https://audioprograming.wordpress.com/2012/03/02/a-phase-vocoder-in-python/
def length_shift(signalin, tscale):
  L = len(signalin)
  # signal blocks for processing and output
  phi  = zeros(N)
  out = zeros(N, dtype=complex)
  sigout = zeros(L/tscale+N)

  # max input amp, window
  amp = max(signalin)
  win = hanning(N)
  p = 0
  pp = 0

  while p < L-(N+H):

    # take the spectra of two consecutive windows
    p1 = int(p)
    spec1 =  fft(win*signalin[p1:p1+N])
    spec2 =  fft(win*signalin[p1+H:p1+N+H])
    # take their phase difference and integrate
    phi += (angle(spec2) - angle(spec1))
    # bring the phase back to between pi and -pi
    for i in phi:
      while i < -pi: i += 2*pi
      while i >= pi: i -= 2*pi
    out.real, out.imag = cos(phi), sin(phi)
    # inverse FFT and overlap-add
    sigout[pp:pp+N] += (win*ifft(abs(spec2)*out)).real
    pp += H
    p += H*tscale
  return array(amp * sigout / max(sigout))

# Given a signal, creates a signal that is `tscale` times shorter
# with altered pitch.
def length_compress(signalin, tscale):
  adj_len = int(len(signalin) / tscale)
  requested = [float(k) * tscale for k in range(adj_len)]
  points = range(len(signalin))
  return interp(requested, points, signalin)

# Given a signal, creates a signal with an altered pitch
# without changing the length.
def pitch_shift(signalin, num_half_tones):
  adjust = pow(2, float(num_half_tones) / 12.)
  compressed = length_compress(signalin, adjust)
  return length_shift(compressed, 1. / adjust)
