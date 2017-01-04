# convNN_CIFAR10  

A simple convolutional neural network for CIFAR10 dataset in tensorflow   

Files:  
main.py - Includes the final training and testing code  
model.py - Includes just the model skeleton (i.e. the neural network)  
train.py - Includes the loss function, training and evaluation code  
helper_data.py - Includes data preprocessing code and the data batch class  
helper_nn.py - Includes code required to instantiate the model  
  
Dependencies:  
tensorflow - used r0.11 to write and test the code  
numpy: basic matrix operations  
scipy: for creating image distortions  
argparse: for command-line inputs  
h5py: for reading in the CIFAR10 data in hdf5 format  
