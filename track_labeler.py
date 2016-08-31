import soundfile as sf
import numpy.fft as fft
import matplotlib.pyplot as plt
import pygame

import sys
import os
import random

CLASSIFICATION_DATA_FNAME = 'data.txt'
PITCH_SAMPLE_MAP = { }

should_play_sounds = False
current_sound = 0

def read_classification_data(directory):
  sample_map = { }
  fname = 'raw/%s/%s' % (directory, CLASSIFICATION_DATA_FNAME)
  if not os.path.exists(fname):
    return sample_map
  f = open(fname, 'r')

  for line in f:
    if line == '':
      continue
    sample, pitch = line.split(':')
    sample = sample.strip()
    pitch = pitch.strip()
    sample_map[int(sample)] = pitch

  f.close()
  return sample_map

def write_classification_data(directory, pitch_map):
  fname = 'raw/%s/%s' % (directory, CLASSIFICATION_DATA_FNAME)

  f = open(fname, 'w')
  for key in pitch_map:
    f.write('%d:%s\n' % (key, pitch_map[key]))
  f.close()

def pick_random_sound():
  global current_sound
  global PITCH_SAMPLE_MAP
  current_sound = random.randint(0, num_samples-1)
  if len(PITCH_SAMPLE_MAP.keys()) < num_samples:
    while current_sound in PITCH_SAMPLE_MAP:
      current_sound = random.randint(0, num_samples-1)
  else:
    current_sound = 0
    print 'All data (%d values) labeled!' % num_samples

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print 'Not enough arguments; requires filename.'
  else:
    directory = sys.argv[1]
    PITCH_SAMPLE_MAP = read_classification_data(directory)
    num_samples = 0
    while os.path.exists('raw/%s/%s.wav' % (directory, str(num_samples))):
      num_samples += 1

    pygame.mixer.init()

    if len(PITCH_SAMPLE_MAP.keys()):
      nonx_keys = filter(lambda x: PITCH_SAMPLE_MAP[x] != 'x', PITCH_SAMPLE_MAP.keys())
      print 'Pitch map loaded with %d keys (%d total)!' % (len(nonx_keys), len(PITCH_SAMPLE_MAP.keys()))

    pick_random_sound()
    print 'Ready for input...!'
    while True:
      user_input = raw_input()

      if user_input == 's' or user_input == 'start':
        should_play_sounds = True
      elif user_input == 'q' or user_input == 'quit':
        should_play_sounds = False
        write_classification_data(directory, PITCH_SAMPLE_MAP)
        break
      elif user_input == 'r' or user_input == 'repeat':
        pass
      elif should_play_sounds and len(user_input) > 0:
        if (user_input[0] >= 'a' and user_input[0] <= 'g') or user_input[0] == 'x':
      	  PITCH_SAMPLE_MAP[current_sound] = user_input.lower()
      	  print 'Recorded sample %d as %s.' % (current_sound, user_input.lower())
      	  pick_random_sound()
      else:
        print 'Unrecognized command.'

      if should_play_sounds:
        print '(playing sound with index %d)' % (current_sound,)
        pygame.mixer.music.load('raw/%s/%d.wav' % (directory, current_sound))
        pygame.mixer.music.play()
