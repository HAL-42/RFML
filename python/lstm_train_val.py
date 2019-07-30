import os
import numpy as np
import tensorflow as tf
from lstm_model.parse_data import ParseData
from lstm_model.build_model import BuildModel

kBatchSize = 1024
kLearningRate = 0.001
kNumEpochs = 10

h5_data_path = os.path.join('..', 'data', 'h5data.same_mac')
h5_module_data_path = os.path.join(h5_data_path, 'h5_module_data')
train_test_data_path = os.path.join('..', 'data', 'h5data.same_mac', 'h5_train_test_split')
log_path = os.path.join('.', 'log', 'tf.' + os.path.split(h5_data_path)[1] + '.LSTM.log')

kClassesList = []
for module_data_name in os.listdir(h5_module_data_path):
	kClassesList.append(module_data_name.split('.')[0])

kClassesNum  = len(kClassesList)


if __name__ == '__main__':
	# parse
	data_parser = ParseData(train_test_data_path, kClassesList, kClassesNum)
	train_files_and_labels_np, test_files_and_labels_np = data_parser.split_train_test(train_ratio=0.8)

	# build model
	lstm_model = BuildModel(kClassesNum)
	lstm_model.build()

	loss = lstm_model.loss()
	tf.summary.scalar('loss', loss)

	optimizer = lstm_model.optimizer(loss, kLearningRate)

	accuracy = lstm_model.accuracy()
	tf.summary.scalar('accuracy', accuracy)

	merged = tf.summary.merge_all()

	# init
	iteration = 0
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		# writer
		train_writer = tf.summary.FileWriter(os.path.join(log_path, 'lstm_train'), sess.graph)
		test_writer = tf.summary.FileWriter(os.path.join(log_path, 'lstm_test'), sess.graph)

		# Run the initializer
		sess.run(init)

		for epoch in range(kNumEpochs):
			print('****** Epoch: {}/{} ******'.format(epoch, kNumEpochs))

			total_batch = int(np.ceil(train_files_and_labels_np.shape[0] / kBatchSize))

			# shuffle the training data for each epoch
			np.random.shuffle(train_files_and_labels_np)

			# iteration
			for i in range(total_batch):
				# get corrupted batch using the un-corrupted data_train
				start_idx = i*kBatchSize
				end_idx = (i+1)*kBatchSize
				batch_X, batch_Y = data_parser.get_actual_data_and_labels(train_files_and_labels_np[start_idx:end_idx])

				if iteration % 5 == 0:
					train_summary, current_loss, current_accuracy = sess.run([merged, loss, accuracy], feed_dict={lstm_model.X: batch_X, lstm_model.Y: batch_Y})
					train_writer.add_summary(train_summary, iteration)
					print('({}/{}) loss: {}, accuracy: {}'.format(i, total_batch, current_loss, current_accuracy))

					random_idx = np.random.choice(test_files_and_labels_np.shape[0], kBatchSize)
					test_X, test_Y = data_parser.get_actual_data_and_labels(test_files_and_labels_np[random_idx])
					test_summary = sess.run([merged], feed_dict={lstm_model.X: test_X, lstm_model.Y: test_Y})
					test_writer.add_summary(test_summary, iteration)

				_ = sess.run([optimizer], feed_dict={lstm_model.X: batch_X, lstm_model.Y: batch_Y})

				iteration += 1

		train_writer.close()
		test_writer.close()
