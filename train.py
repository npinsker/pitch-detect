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

#np.savetxt('sound_data.out', sound, delimiter=',')
#np.savetxt('label_data.out', label, delimiter=',')

#sound = np.loadtxt('sound_data.out', delimiter=',')
#label = np.loadtxt('label_data.out', delimiter=',')
print 'loaded data'
print sound.shape

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

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

x = tf.placeholder(tf.float32, [None, SAMPLE_SIZE])

LAYER1_SIZE = 128

W1 = weight_variable([SAMPLE_SIZE, LAYER1_SIZE])
b1 = bias_variable([LAYER1_SIZE])

y1 = tf.nn.relu(tf.matmul(x, W1) + b1)

keep_prob = tf.placeholder(tf.float32)
y1_dropout = tf.nn.dropout(y1, keep_prob)

W2 = weight_variable([LAYER1_SIZE, 12])
b2 = bias_variable([12])

y = tf.matmul(y1_dropout, W2) + b2

y_ = tf.placeholder(tf.float32, [None,12])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)

train_step = tf.train.AdamOptimizer(4e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print 'starting loop'

for i in range(10000):
  test_indices = np.random.choice(len(sound), 100)
  test_x = np.array([sound[k] for k in test_indices])
  test_y_ = np.array([label[k] for k in test_indices])
  sess.run(train_step, feed_dict={x: test_x, y_: test_y_, keep_prob: 0.5})
  if i % 100 == 99:
    print 'step %d' % (i+1)
    print 'accuracy (train set):', sess.run(accuracy, feed_dict={x: test_x, y_: test_y_, keep_prob: 1.0})
    if i % 10000 == 9999:
      print 'accuracy (test  set):', sess.run(accuracy, feed_dict={x: test_sound, y_: test_label, keep_prob: 1.0})
    print sess.run(tf.argmax(y,1), feed_dict={x: test_sound, y_: test_label, keep_prob: 1.0})

save_path = saver.save(sess, 'model.ckpt')
print 'Model saved to file: %s' % save_path

