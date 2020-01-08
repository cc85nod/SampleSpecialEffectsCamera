import random
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from load_dataset import load_dataset, resize_image, IMAGE_SIZE

MODEL_PATH = "face_detect_model/jerry.face.model.h5"
# Ignore warning
tf.logging.set_verbosity(tf.logging.ERROR)

class Dataset:
	def __init__(self, path_name):
		"""
		Split to three data categories:
		1. train
		2. valid
		3. test
		"""
		self.train_images = None
		self.train_labels = None

		self.valid_images = None
		self.valid_labels = None

		self.test_images = None
		self.test_labels = None

		self.path_name = path_name

		self.input_shape = None

	def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,
			img_channels=3, nb_classes=2):
		
		# Load the dataset and split data
		images, labels = load_dataset(self.path_name)
		train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.3, random_state=random.randint(0, 100))
		_, test_images, _, test_labels = train_test_split(images, labels, test_size=0.5, random_state=random.randint(0, 100))

		# th => (channels, rows, cols)
		if K.image_dim_ordering() == "th":
			train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
			valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
			test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
			self.input_shape(img_channels, img_rows, img_cols)
		# tf => (rows, cols, channels)
		else:
			train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
			valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
			test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
			self.input_shape = (img_rows, img_cols, img_channels)
		
		print(train_images.shape[0], "train samples")
		print(valid_images.shape[0], "valid samples")
		print(test_images.shape[0], "test samples\n")

		# We will use categorical_crossentropy as loss function
		train_labels = np_utils.to_categorical(train_labels, nb_classes)
		valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
		test_labels = np_utils.to_categorical(test_labels, nb_classes)

		# Convert int to float as it can be normalized
		train_images = train_images.astype("float32")
		valid_images = valid_images.astype("float32")
		test_images = test_images.astype("float32")

		train_images /= 255
		valid_images /= 255
		test_images /= 255

		self.train_images = train_images
		self.valid_images = valid_images
		self.test_images = test_images
		self.train_labels = train_labels
		self.valid_labels = valid_labels
		self.test_labels = test_labels

# CNN
class Model:
	def __init__(self):
		self.mode = None
	
	def build_model(self, dataset, nb_classes=2):
		self.model = Sequential()

		self.model.add(Convolution2D(32, 3, 3, border_mode="same", input_shape=dataset.input_shape))
		self.model.add(Activation("relu"))

		self.model.add(Convolution2D(32, 3, 3))
		self.model.add(Activation("relu"))

		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))

		self.model.add(Convolution2D(64, 3, 3, border_mode="same"))
		self.model.add(Activation("relu"))

		self.model.add(Convolution2D(64, 3, 3))
		self.model.add(Activation("relu"))

		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))

		self.model.add(Flatten())
		self.model.add(Dense(512))
		self.model.add(Activation("relu"))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(nb_classes))
		self.model.add(Activation("softmax"))

		# Output summary of model
		self.model.summary()

	def train(self, dataset, batch_size=20, nb_epoch=10, data_augmentation=True):
		"""
		SGD optimizers
		lr: learning rate
		"""
		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

		# Compile kernel with categorical_crossentropy as loss function, sgd as optimizer
		self.model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
		
		# Basical training
		if not data_augmentation:
			"""
			Explanation:
			batch_size => how many datas per calculation
			epochs => traing period
			"""
			self.model.fit(
				dataset.train_images,
				dataset.train_labels,
				batch_size=batch_size,
				epochs=nb_epoch,
				validation_data=(dataset.valid_images, dataset.valid_labels),
				shuffle=True
			)
		else:
			"""
			Explanation:
			> data preprocessor
			featurewise_center => set input mean to 0 over the dataset(data decentered)
			samplewise_center => set image mean to 0
			featurewise_std_normalization => feature normalized
			samplewise_std_normalization => sample normalized
			zca_whitening => ZCA whiten(use PCA)
			rotation_range => image rotate degree
			"""
			datagen = ImageDataGenerator(
				featurewise_center=False,
				samplewise_center=False,
				featurewise_std_normalization=False,
				samplewise_std_normalization=False,
				zca_whitening=False,
				rotation_range=20,
				width_shift_range=0.2,
				height_shift_range=0.2,
				horizontal_flip=True,
				vertical_flip=False,
			)

			datagen.fit(dataset.train_images)

			self.model.fit_generator(
				datagen.flow(dataset.train_images, dataset.train_labels, batch_size=batch_size),
				samples_per_epoch=dataset.train_images.shape[0],
				nb_epoch=nb_epoch,
				validation_data=(dataset.valid_images, dataset.valid_labels)
			)

	def save_model(self, file_path=MODEL_PATH):
		self.model.save(file_path)

	def load_model(self, file_path=MODEL_PATH):
		self.model = load_model(file_path)

	def evaluate(self, dataset):
		score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
		print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))

	def face_predict(self, image):
		if K.image_dim_ordering() == "th" and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
			image = resize_image(image)
			image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))
		elif K.image_dim_ordering() == "tf" and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
			image = resize_image(image)
			image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

		image = image.astype("float32")
		image /= 255

		result = self.model.predict_proba(image)
		print("result: ", result)

		result = self.model.predict_classes(image)

		return result[0]

def main():
	dataset = Dataset("face_samples/")
	dataset.load()

	model = Model()
	model.build_model(dataset)

	model.train(dataset)

	model.save_model()
	# model.load_model()
	
	model.evaluate(dataset)

if __name__=="__main__":
	main()
