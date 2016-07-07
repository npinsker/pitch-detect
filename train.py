import soundfile as sf
import numpy as np
import numpy.fft as fft
import data_reader
import utilities
import tensorflow as tf

SAMPLE_SIZE = 4410

sound, label = data_reader.read_data_for_track('grace')
test_sound, test_label = data_reader.read_data_for_track('bach')

print 'pitch shifting ', len(label), ' input entries'

sound_arr = [ ]
label_arr = [ ]

for i in range(1, 12):
  new_sound, new_label = data_reader.shift_data_block_pitch(sound, label, i)
  sound_arr += [new_sound]
  label_arr += [new_label]
for i in range(11):
  sound = np.vstack((sound, sound_arr[i]))
  label = np.vstack((label, label_arr[i]))

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


train_step = tf.train.AdagradOptimizer(1.).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print 'starting loop'

for i in range(4000):
  sess.run(train_step, feed_dict={x: sound, y_: label})
  if i % 100 == 99:
    print 'step %d' % (i+1)
    print 'accuracy (train set):', sess.run(accuracy, feed_dict={x: sound, y_: label})
    print 'accuracy (test  set):', sess.run(accuracy, feed_dict={x: test_sound, y_: test_label})
    print sess.run(tf.argmax(y,1), feed_dict={x: test_sound, y_: test_label})
print b.eval(sess)


print sess.run(accuracy, feed_dict={x: test_sound, y_: test_label})
