import pandas as pd
import idx2numpy
#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

#Display image 
def display(image,label):
    """image is a 1*784 numpy array"""
    image = image.reshape(28,28)
    plt.imshow(image, cmap = plt.cm.gray_r, interpolation="nearest")
    plt.title("Image Representation for %d" %(label))
    plt.show()

#Extract Training set
X_train_3D = idx2numpy.convert_from_file('train-images.idx3-ubyte')
X_train = X_train_3D.flatten().reshape(60000,784)

y_train = idx2numpy.convert_from_file('train-labels.idx1-ubyte').reshape(60000,1)

#Extract Test set
X_test_3D = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
X_test =  X_test_3D.flatten().reshape(10000,784)

y_test = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte').reshape(10000,1)

#print sample images 
for i in range(10):
    display(X_train[i],y_train[i][0])