import os
import numpy as np
import tensorflow as tf
from lstm_model.build_model import BuildModel
from lstm_model.data_manager import DataManager

kBatchSize = 1024
kLearningRate = 0.001
kNumEpochs = 10

kH5DataPath = os.path.join('..', 'data', 'h5data.same_mac')
kH5ModuleDataPath = os.path.join(kH5DataPath, 'h5_module_data')
kH5TrainTestDataPath = os.path.join(kH5DataPath, 'h5_train_test_split')
kLogPath = os.path.join('.', 'log', 'tf.' + os.path.split(kH5DataPath)[1] + '.LSTM.log')
kSnapshotPath = os.path.join(kLogPath, 'snapshot', 'LSTM')


if __name__ == '__main__':
	# parse
	data_manager = DataManager(kH5TrainTestDataPath, kH5ModuleDataPath, I_only=True, down_sample=0)

	# build model
	lstm_model = BuildModel(data_manager.classes_num)
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

	saver = tf.train.Saver(max_to_keep=5)
	with tf.Session() as sess:
		# writer
		train_writer = tf.summary.FileWriter(os.path.join(kLogPath, 'lstm_train'), sess.graph)
		test_writer = tf.summary.FileWriter(os.path.join(kLogPath, 'lstm_test'), sess.graph)

		# Run the initializer
		sess.run(init)
		saver.save(sess, kSnapshotPath)
		for epoch in range(kNumEpochs):
			print('****** Epoch: {}/{} ******'.format(epoch, kNumEpochs))
			batches_num = int(np.ceil(data_manager.train_samples_num / kBatchSize))
			# Init data_manager
			data_manager.init_epoch()
			# Get batches generator
			train_batches = data_manager.get_train_batches(kBatchSize)

			# iteration
			for i, train_batch in enumerate(train_batches):
				# get corrupted batch using the un-corrupted data_train
				batch_X, batch_Y = train_batch
				batch_X = batch_Y.reshape(batch_X.shape[0], -1, lstm_model.TIMESTEPS)

				if iteration % 5 == 0:
					train_summary, current_loss, current_accuracy = sess.run([merged, loss, accuracy], feed_dict={lstm_model.X: batch_X, lstm_model.Y: batch_Y})
					train_writer.add_summary(train_summary, iteration)
					print('({}/{}) loss: {}, accuracy: {}'.format(i, batches_num, current_loss, current_accuracy))

					test_X, test_Y = data_manager.get_random_test_samples(kBatchSize)
					test_X = test_X.reshape(test_X.shape[0], -1, lstm_model.TIMESTEPS)
					test_summary = sess.run([merged], feed_dict={lstm_model.X: test_X, lstm_model.Y: test_Y})
					test_writer.add_summary(test_summary, iteration)

				_ = sess.run([optimizer], feed_dict={lstm_model.X: batch_X, lstm_model.Y: batch_Y})

				iteration += 1
			saver.save(sess, kSnapshotPath, global_step=epoch)
		train_writer.close()
		test_writer.close()
