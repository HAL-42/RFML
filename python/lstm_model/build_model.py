import tensorflow as tf

class BuildModel():
	TOTAL_SAMPLE_LENGTH = 10000
	TIMESTEPS = 50
	NUM_INPUT = int(TOTAL_SAMPLE_LENGTH / TIMESTEPS)
	NUM_HIDDEN = 1024

	def __init__(self, num_classes):
		self.num_classes = num_classes

		self._create_tensors()

	def _create_tensors(self):
		self.X = tf.placeholder('float32', [None, self.TIMESTEPS, self.NUM_INPUT])
		self.Y = tf.placeholder('int8', [None, self.num_classes])

		# weights and biases
		self.weights = {
			'out': tf.Variable(tf.random_normal([self.NUM_HIDDEN, self.num_classes]))
		}

		self.biases = {
			'out': tf.Variable(tf.random_normal([self.num_classes]))
		}

	def build(self):
		# Prepare data shape to match `rnn` function requirements
		# Current data input shape: (batch_size, timesteps, n_input)
		# Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

		# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
		x = tf.unstack(self.X, self.TIMESTEPS, 1)

		# Define a lstm cell with tensorflow
		lstm_cell = tf.nn.rnn_cell.LSTMCell(self.NUM_HIDDEN, forget_bias=1.0)

		# Get lstm cell output
		outputs, _ = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)

		# Linear activation, using rnn inner loop last output
		self.logits = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']
		self.prediction = tf.nn.softmax(self.logits)

	def loss(self):
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))

	def optimizer(self, loss, learning_rate):
		return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

	def accuracy(self):
		# Evaluate model (with test logits, for dropout to be disabled)
		correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
		return tf.reduce_mean(tf.cast(correct_pred, tf.float32))
