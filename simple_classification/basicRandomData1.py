import numpy as np
from itertools import repeat
import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)

###Test parameters:
sample_width = 5
nb_train_samples = 20000
nb_test_samples = 1000

###Making the layers:
labels = tf.placeholder(tf.float32, shape=(None,1))
features = tf.placeholder(tf.float32, shape=(None,sample_width))

from keras.models import Sequential
from keras.layers import Dense
import random

model = Sequential()
first_layer = Dense(20, activation='sigmoid', input_shape=(None,sample_width))
first_layer.set_input(features)
model.add(first_layer)
model.add(Dense(1, activation='sigmoid'))
output_layer = model.output

###making training data & test data:

train_features = np.random.randn(nb_train_samples, sample_width)
train_labels = np.zeros(nb_train_samples).reshape(nb_train_samples, 1)
test_features = np.random.randn(nb_test_samples, sample_width)
test_labels = np.zeros(nb_test_samples).reshape(nb_test_samples, 1)

train_ones = 0
test_ones = 0

for i in range(nb_train_samples):
	if(random.random() < 0.5):#50 % of the time make feature[2] = 1 & the output also = 1
		train_features[i, 2] = 1
		train_labels[i] = 1
		train_ones = train_ones + 1
	else:
		train_features[i,2] = 0
print "Expect "+str(train_ones)+" ones in training set"

for i in range(nb_test_samples):
	if(random.random() < 0.5):#50 % of the time make feature[2] = 1 & the output also = 1
		test_features[i, 2] = 1
		test_labels[i] = 1
		test_ones = test_ones + 1
	else:
		test_features[i,2] = 0
print "Expect "+str(test_ones)+" ones in test set"

###Objective function:

#from keras.objectives import categorical_crossentropy
#loss = tf.reduce_mean(categorical_crossentropy(labels, output_layer))

from keras.objectives import mean_squared_error
loss = tf.reduce_mean(mean_squared_error(labels, output_layer))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

###training:
with sess.as_default():
	for i in range(nb_train_samples): #Batching this might help also.
		train_step.run(feed_dict={
			features:train_features[i,:].reshape(1,sample_width), 
			labels: train_labels[i,:].reshape(1,1),
			K.learning_phase(): 1
		})


###evaluation:
correct_prediction = tf.equal(tf.round(output_layer), labels)
prediction = output_layer

accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

with sess.as_default():
	print "Test running:"
	print  sess.run(prediction,  feed_dict={
			features: test_features,
			labels:test_labels,
			K.learning_phase(): 0
		})

	print "\nTest accuracy:"
	print sess.run(accuracy, feed_dict={
			features: test_features,
			labels:test_labels,
			K.learning_phase(): 0
		})





