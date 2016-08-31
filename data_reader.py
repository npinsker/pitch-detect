import soundfile as sf
import numpy as np
import numpy.fft as fft
import track_labeler
import utilities

def read_data_with_flat_pitch(data_file, pitch):
  sound = sf.read(data_file)[0]
  label_index = np.eye(12)[utilities.note_name_to_index(pitch)]
  if isinstance(sound[0], np.ndarray):
    sound = np.array([(k[0] + k[1]) / 2 for k in sound])
  num_to_process = int(len(sound) / 4410)
  out_sound = np.ndarray(shape=(num_to_process,4410))
  out_label = np.ndarray(shape=(num_to_process,12))
  for p in range(0, len(sound)-4410, 4410):
    out_sound[int(p / 4410)] = sound[p:p+4410]
    out_label[int(p / 4410)] = label_index
  return out_sound, out_label

# Reads specially formatted data in directory given by 'directory'.
# The directory should contain WAV files of length 0.1s (4,410 samples)
# labeled 0.wav, ..., N.wav.
# The directory should also contain a data file 'data.wav' which
# contains key-value pairs assigning each WAV file to a pitch.
# See track-labeler.py for more information.
def read_data_for_track(directory):
  data_map = track_labeler.read_classification_data(directory)
  out_sound = np.ndarray(shape=(0,4410))
  out_label = np.ndarray(shape=(0,12))
  for key in data_map:
    if data_map[key] == 'x':
      continue
    sound = sf.read('raw/%s/%s.wav' % (directory, key))[0]
    if isinstance(sound[0], np.ndarray):
      sound = np.array([(k[0] + k[1]) / 2 for k in sound])
    label_index = utilities.note_name_to_index(data_map[key])
    out_sound = np.vstack((out_sound, sound))
    out_label = np.vstack((out_label, np.eye(12)[label_index]))
  return out_sound, out_label

def read_data_for_tracks(directories):
  out_sound = np.ndarray(shape=(0,4410))
  out_label = np.ndarray(shape=(0,12))
  for directory in directories:
    dir_sound, dir_label = read_data_for_track(directory)
    out_sound = np.vstack((out_sound, dir_sound))
    out_label = np.vstack((out_label, dir_label))
  return out_sound, out_label

def shift_data_block_pitch(sound, label, num_half_tones):
  out_sound = np.ndarray(shape=(0,4410))
  out_label = np.ndarray(shape=(0,12))
  for i in range(len(sound)):
    out_sound = np.vstack((out_sound, utilities.pitch_shift(sound[i], num_half_tones)))
    out_label = np.vstack((out_label, np.concatenate((label[i,-num_half_tones:], label[i,:-num_half_tones]))))
  return out_sound, out_label
