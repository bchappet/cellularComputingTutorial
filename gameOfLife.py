import numpy as np
class GameOfLife():
    """
    Simple class to hold the state of the cellular automata
    """
    def __init__(self,shape):
        self.shape = shape
        self._pointer = False
        self._buffers = [self._new_buffer(),self._new_buffer()]


    def reset(self):
        """
        re initialization of the buffers
        """
        self._buffers = [self._new_buffer(),self._new_buffer()]

    def _new_buffer(self):
        return np.zeros(self.shape)
                
    
    def set(self,array):
        self._buffers[int(self._pointer)][...] = array
        
    def get(self):
        return self._buffers[int(self._pointer)]
        
    def run(self,nbIteration=1):
        for i in range(nbIteration):
            self.array = self._step(self._buffers[int(self._pointer)],self._buffers[int(not(self._pointer))])
            self._pointer = not(self._pointer)
        
    def _step(self,X,Y):
        for i in range(1,X.shape[0]-1):
            for j in range(1,X.shape[1]-1):
                
                # Count neighbors
                N = (X[i-1,j-1] + X[i-1,j] + X[i-1,j+1] +
                     X[i  ,j-1]            + X[i  ,j+1] +
                     X[i+1,j-1] + X[i+1,j] + X[i+1,j+1])

                # Apply rules
                state = X[i,j]
                if state:
                    if((N==2) | (N==3)):
                        newState = 1
                    else:
                        newState = 0
                else:
                    if N == 3:
                        newState = 1
                    else:
                        newState = 0
                             
                
                #Update new array
                Y[i,j] = newState
