import soundfile as sf
import numpy as np
import numpy.fft as fft
import track_labeler
import utilities

def read_data_for_track(directory):
  data_map = track_labeler.read_classification_data(directory)
  out_sound = np.ndarray(shape=(0,4410))
  out_label = np.ndarray(shape=(0,12))
  for key in data_map:
    if data_map[key] == 'x':
      continue
    sound = sf.read('raw/%s/%s.wav' % (directory, key))[0]
    if isinstance(sound, np.ndarray):
      sound = np.array([(k[0] + k[1]) / 2 for k in sound])
    label = utilities.note_name_to_index(data_map[key])
    out_sound = np.vstack((out_sound, sound))
    out_label = np.vstack((out_label, np.eye(12)[label]))
  return out_sound, out_label

def shift_data_block_pitch(sound, label, num_half_tones):
  out_sound = np.ndarray(shape=(0,4410))
  out_label = np.ndarray(shape=(0,12))
  for i in range(len(sound)):
    out_sound = np.vstack((out_sound, utilities.pitch_shift(sound[i], num_half_tones)))
    out_label = np.vstack((out_label, np.concatenate((label[i,-num_half_tones:], label[i,:-num_half_tones]))))
  return out_sound, out_label
