from gameOfLife import GameOfLife
import numpy as np

class GameOfLifeBool(GameOfLife):
        
    def _new_buffer(self):
        return np.zeros(self.shape,dtype=np.bool_)
        
        
    def _step(self,X,Y):
        X_int = X.astype(np.uint8) #old array converted into uint
        # Count neighbors
        N = (X_int[0:-2,0:-2] + X_int[0:-2,1:-1] + X_int[0:-2,2:] +
             X_int[1:-1,0:-2] +                    X_int[1:-1,2:] +
             X_int[2:  ,0:-2] + X_int[2:  ,1:-1] + X_int[2:  ,2:])
        # Apply rules 
        birth = (N==3) & (~X[1:-1,1:-1]) 
        survive = ((N==2) | (N==3)) & (X[1:-1,1:-1]) 
        #reset the cells
        Y[...] = 0
        #switch on the cells with the rules boolean array
        Y[1:-1,1:-1] = birth | survive
