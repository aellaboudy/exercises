from keras.datasets import cifar10
import numpy


#load the whole cifar-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print numpy.shape(x_train)
print numpy.shape(y_train)

semi_output = numpy.zeros([4000,1]) 
semi_input = numpy.zeros([4000,32,32,3])

# Load 400 of each label to make sure dataset is balanced
for i in range(0,10):
	indices = numpy.where(y_train == i)[0]
	indices = indices[0:400]
	semi_output[i*400:400+i*400,...] = y_train[indices,...]
	semi_input[i*400:400+i*400,...] = x_train[indices,...]

print numpy.shape(semi_input)
print numpy.shape(semi_output)

#verify that there are equal occurances of each label...
unique, counts = numpy.unique(semi_output, return_counts=True)
print (dict(zip(unique, counts)))





