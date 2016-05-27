#see https://github.com/greenelab/gcb535challenge for corresponding data.

import numpy as np
from itertools import repeat
import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)

###Making the layers:
labels = tf.placeholder(tf.float32, shape=(None,1))
features = tf.placeholder(tf.float32, shape=(None,200))

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1l2, activity_l1l2
import random

learning_rate = 0.1001

model = Sequential()
first_layer = Dense(95, activation='sigmoid', input_shape=(None,200),
			W_regularizer=l1l2(l1=0.1, l2=0.),
			activity_regularizer=activity_l1l2(l1=0.1, l2=0.))
first_layer.set_input(features)
model.add(first_layer)
model.add(Dense(1, activation='sigmoid'))
output_layer = model.output

###making training data & test data:
train_data = np.loadtxt(open("data/D1_S1.csv", "rb"), delimiter=",")
train_features = train_data[:, :200]
train_labels = train_data[:, 200]

test_data = np.loadtxt(open("data/D1_S2.csv", "rb"), delimiter=",")
test_features = test_data[:, :200]
test_labels = test_data[:, 200]

###Objective function:
from keras.objectives import mean_squared_error
loss = tf.reduce_mean(mean_squared_error(labels, output_layer))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

###training:
with sess.as_default():
	for i in range(train_labels.size):
		train_step.run(feed_dict={
			features:train_features[i].reshape(1,200),
			labels: np.asarray(train_labels[i]).reshape(1,1),
			K.learning_phase(): 1
		})


###evaluation:
correct_prediction = tf.equal(tf.round(output_layer), labels)
prediction = output_layer

accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

with sess.as_default():
		
#Note: accuracy of all 0.495/0.505 is all 1/0 response.
	print "learning @"+str(learning_rate)+"->"+str(sess.run(accuracy, feed_dict={
			features: test_features,
			labels:np.asarray(test_labels).reshape(test_labels.size, 1),
			K.learning_phase(): 0
		}))





