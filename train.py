import soundfile as sf
import numpy as np
import numpy.fft as fft
import data_reader
import utilities
import tensorflow as tf

SAMPLE_SIZE = 4410

sound, label = data_reader.read_data_for_track('downstream')
test_sound, test_label = data_reader.read_data_for_track('arise')

print len(label)

sound_arr = [ ]
label_arr = [ ]

for i in range(1, 12):
  new_sound, new_label = data_reader.shift_data_block_pitch(sound, label, i)
  sound_arr += [new_sound]
  label_arr += [new_label]
for i in range(11):
  sound = np.vstack((sound, sound_arr[i]))
  label = np.vstack((label, label_arr[i]))
for i in range(len(sound)):
  sound[i] = fft.fft(sound[i]).real
for i in range(len(test_sound)):
  test_sound[i] = fft.fft(test_sound[i]).real

print 'data pitch shifting finished'
print sound.shape

x = tf.placeholder(tf.float32, [None, SAMPLE_SIZE])

W = tf.Variable(tf.zeros([SAMPLE_SIZE, 12]))
b = tf.Variable(tf.zeros([12]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None,12])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
  sess.run(train_step, feed_dict={x: sound, y_: label})
  if i % 100 == 99:
    print 'step %d' % (i+1)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print sess.run(accuracy, feed_dict={x: test_sound, y_: test_label})
