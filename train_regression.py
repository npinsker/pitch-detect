import soundfile as sf
import numpy as np
import numpy.fft as fft
import data_reader
import utilities
import tensorflow as tf
import random

SAMPLE_SIZE = 4410

sound = np.ndarray(shape=(0,SAMPLE_SIZE))
label = np.ndarray(shape=(0,12))

pitches = ["a4", "a#4", "b4", "c5", "c#5", "d5", "d#5", "e5", "f5", "f#5", "g5", "g#5"]
for i in range(12):
  print 'loading pitch %d' % i
  new_sound, new_label = data_reader.read_data_with_flat_pitch('raw/a%d.wav' % i, pitches[i])
  sound = np.vstack((sound, new_sound))
  label = np.vstack((label, new_label))

print sound.shape
print label.shape
sound_arr = [ ]
label_arr = [ ]
print 'shuffling sound array'

for i in range(10*len(sound)):
  j, k = random.randint(0, len(sound)-1), random.randint(0, len(sound)-1)
  tmp, tmp_label = sound[j], label[j]
  sound[j] = sound[k]
  label[j] = label[k]
  sound[k] = tmp
  label[k] = tmp_label

test_data_size = len(sound) * 9 / 10
test_sound = sound[test_data_size:]
test_label = label[test_data_size:]
sound = sound[:test_data_size]
label = label[:test_data_size]

print 'data pitch shifting finished'

for i in range(len(sound)):
  sound[i] = fft.fft(sound[i]).real
for i in range(len(test_sound)):
  test_sound[i] = fft.fft(test_sound[i]).real

x = tf.placeholder(tf.float32, [None, SAMPLE_SIZE])

W = tf.Variable(tf.zeros([SAMPLE_SIZE, 12]))
b = tf.Variable(tf.zeros([12]))

y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None,12])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)

train_step = tf.train.AdagradOptimizer(0.02).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print 'starting loop'

for i in range(40000):
  test_indices = np.random.choice(len(sound), 100)
  test_x = np.array([sound[k] for k in test_indices])
  test_y_ = np.array([label[k] for k in test_indices])
  sess.run(train_step, feed_dict={x: test_x, y_: test_y_})
  if i % 10 == 9:
    print 'step %d' % (i+1)
    print 'accuracy (train set):', sess.run(accuracy, feed_dict={x: test_x, y_: test_y_})
    print 'accuracy (test  set):', sess.run(accuracy, feed_dict={x: test_sound, y_: test_label})
    print sess.run(tf.argmax(y,1), feed_dict={x: test_sound, y_: test_label})
print b.eval(sess)


print sess.run(accuracy, feed_dict={x: test_sound, y_: test_label})
