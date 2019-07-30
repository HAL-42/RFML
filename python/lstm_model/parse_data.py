import os
import numpy as np
import pandas as pd

class ParseData():
	def __init__(self, data_parent_dir, class_list, num_classes):
		self.data_parent_dir = data_parent_dir
		self.class_list = class_list
		self.num_classes = num_classes

		self.total_num_files, self.all_files_and_labels_np = self._get_file_names_and_labels()

	def split_train_test(self, train_ratio):
		np.random.shuffle(self.all_files_and_labels_np)

		num_train_files = int(self.total_num_files * train_ratio)
		num_test_files = self.total_num_files - num_train_files

		print('number of training files: {}'.format(num_train_files))
		print('number of testing files: {}'.format(num_test_files))

		train_files_and_labels_np = self.all_files_and_labels_np[0:num_train_files]
		test_files_and_labels_np = self.all_files_and_labels_np[num_train_files:]

		return train_files_and_labels_np, test_files_and_labels_np

	def _get_file_names_and_labels(self):
		# get number of files per each class
		num_files_per_class_dict = {}

		for class_name in self.class_list:
			data_class_dir = os.path.join(self.data_parent_dir, 'data_{}'.format(class_name))
			num_files = len([fn for fn in os.listdir(data_class_dir) if os.path.isfile(os.path.join(data_class_dir, fn))])

			num_files_per_class_dict[class_name] = num_files

		print('number of files per class: {}'.format(num_files_per_class_dict))

		# get total number of files
		total_num_files = 0

		for _, num_files in num_files_per_class_dict.items():
			total_num_files += num_files

		print('total number of files: {}'.format(total_num_files))

		# fill up numpy array to contain all files and labels
		count = 0
		all_files_and_labels_np = np.empty((total_num_files,2), dtype=object)

		for class_name, num_files in num_files_per_class_dict.items():
			data_class_dir = os.path.join(self.data_parent_dir, 'data_{}'.format(class_name))

			for i in range(num_files):
				filename = os.path.join(data_class_dir, '{}_{}.csv'.format(class_name, i+1))
				all_files_and_labels_np[count, 0] = filename
				all_files_and_labels_np[count, 1] = class_name
				count += 1

		assert count == total_num_files

		return total_num_files, all_files_and_labels_np

	def get_actual_data_and_labels(self, train_files_and_labels_np):
		batch_size = train_files_and_labels_np.shape[0]
		train_file_names = train_files_and_labels_np[:, 0]
		train_labels = train_files_and_labels_np[:, 1]

		x_batch = np.empty((batch_size, 40000), dtype=np.float32)
		y_batch = np.empty((batch_size, self.num_classes), dtype=np.int8)

		y_label = []
		size_incompatible_list = []

		for i in range(batch_size):
			df = pd.read_csv(train_file_names[i], header=None, names=['I', 'Q'])
			df_values = df.loc[:, 'I'].values
			if df_values.shape[0] != 40000:
				size_incompatible_list.append(i)
				x_batch[i] = np.zeros((40000,))
			else:
				x_batch[i] = df_values

			y_label.append(train_labels[i])

		x_batch = np.delete(x_batch, size_incompatible_list, axis=0)
		x_batch = x_batch[:, 0::4] # downsize
		x_batch = np.reshape(x_batch, (-1, 50, 200))

		y_batch = pd.get_dummies(y_label).values
		y_batch = np.delete(y_batch, size_incompatible_list, axis=0)

		return x_batch, y_batch
