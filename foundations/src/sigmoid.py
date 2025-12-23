'''
Sigmoid Activation Function
'''
import numpy as np

def sigmoid(z:int):
    ''' 
    Sigmoid activation function.
    '''
    return 1 / (1 + np.exp(-z))

if __name__ == "__main__":
  var1 = 6
  print(sigmoid(var1))