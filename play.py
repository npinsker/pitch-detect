import alsaaudio, time, audioop
import soundfile as sf
import numpy
import numpy.fft as fft
import tensorflow as tf

inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK)

inp.setchannels(1)
inp.setrate(44100)
inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)

inp.setperiodsize(160)

all_data = numpy.array([ ], dtype='int16')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

SAMPLE_SIZE = 4410
LAYER1_SIZE = 128

x = tf.placeholder(tf.float32, [None, SAMPLE_SIZE])

W1 = weight_variable([SAMPLE_SIZE, LAYER1_SIZE])
b1 = bias_variable([LAYER1_SIZE])

y1 = tf.nn.relu(tf.matmul(x, W1) + b1)

keep_prob = tf.placeholder(tf.float32)
y1_dropout = tf.nn.dropout(y1, keep_prob)

W2 = weight_variable([LAYER1_SIZE, 12])
b2 = bias_variable([12])

y = tf.matmul(y1_dropout, W2) + b2

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, 'model.ckpt')
print 'Model restored!'

while True:
  l, data = inp.read()
  if l:
    a_data = numpy.fromstring(data, dtype='int16')
    #adj_data = numpy.array([(a_data[2*i] + a_data[2*i+1]) / 2 for i in range(len(a_data)/2)],
    #                       dtype='int16')
    adj_data = a_data
    all_data = numpy.concatenate((all_data, adj_data))
    if len(all_data) >= SAMPLE_SIZE:
      transformed_data = fft.fft(all_data[:SAMPLE_SIZE]).real.reshape((1,4410))
      result = sess.run(tf.argmax(y,1), feed_dict={x: transformed_data, keep_prob: 1.0})
      pitches = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]
      print pitches[result], ',', result
      all_data = numpy.array([ ], dtype='int16')
  time.sleep(0.001)
