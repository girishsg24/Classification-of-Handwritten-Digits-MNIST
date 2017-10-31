import pandas as pd
import idx2numpy

X_train_3D = idx2numpy.convert_from_file('train-images.idx3-ubyte')
X_train = X_train_3D.flatten().reshape(60000,784)

y_train = idx2numpy.convert_from_file('train-labels.idx1-ubyte').reshape(60000,1)

X_test_3D = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
X_test =  X_test_3D.flatten().reshape(10000,784)

y_test = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte').reshape(10000,1)





y = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
X_flatten = X.flatten()
X_flatten = X_flatten.reshape(60000,784)



