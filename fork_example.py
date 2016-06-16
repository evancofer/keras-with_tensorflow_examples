#Simple example showing how the merge layer works.

from keras.layers import Convolution1D, MaxPooling1D, merge, UpSampling1D
def buildFork(inputs):
	fork0 = Convolution1D(64, filter_length=1, activation='relu', border_mode='same')(inputs)

	fork1 = Convolution1D(48, filter_length=1, activation='relu', border_mode='same')(inputs)
	fork1 = Convolution1D(64, filter_length=3, activation='relu', border_mode='same')(fork1)
	fork1 = Convolution1D(80, filter_length=3, activation='relu', border_mode='same')(fork1)

	fork2 = Convolution1D(80, filter_length=1, activation='relu', border_mode='same')(inputs)
	fork2 = Convolution1D(112, filter_length=3, activation='relu', border_mode='same')(fork2)

	fork3 = MaxPooling1D(pool_length=2)(inputs)
	fork3 = UpSampling1D(length=2)(fork3) #TODO (Evan): Remove this once 'same' Pooling border_mode added to Theano backed Keras. Or when uniform zero padding is added.
	fork3 = Convolution1D(32, filter_length=1, activation='relu', border_mode='same')(fork3)

	modelLayers = merge([fork0, fork1, fork2, fork3], mode='concat') #Note (Evan): I was unable to use the "Merge" layer. The "merge" layer works fine however.

	modelLayers = MaxPooling1D(pool_length=2)(modelLayers)
	return ModelLayers
