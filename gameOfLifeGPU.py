from gameOfLifeBool import GameOfLifeBool 
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

class GameOfLifeGPU(GameOfLifeBool):
    def __init__(self,shape):
        super().__init__(shape)
        self._init_cuda_mem()
        self.init_cuda()
    
    def _init_cuda_mem(self):
        array = self._buffers[int(self._pointer)]
        self._cuda_buffers = [cuda.mem_alloc(array.nbytes),cuda.mem_alloc(array.nbytes)]
        
        thX  = 16
        thY  = 16
        grid_height = max(1,self.shape[0]//thY)
        grid_width = max(1,self.shape[1]//thX)
        
        self._block = (thY,thX,1)
        self._grid = (grid_height,grid_width)
        print(self._grid)
        
    
    def set(self,array):
        super().set(array)
        gpu_array = self._get_gpu()
        cuda.memcpy_htod(gpu_array,array)
    
    def _get_gpu(self):
        return self._cuda_buffers[int(self._pointer)]
            
    def get(self):
        array = super().get()
        gpu_array = self._get_gpu()
        cuda.memcpy_dtoh(array,gpu_array)
        return array
    
        
    
    
    
    def init_cuda(self):
        mod = SourceModule("""
        __global__ void bit_gol(const bool *data, bool *dataRes,
            const uint nbRow,const uint nbCol)
        {
            unsigned int x = blockIdx.x * blockDim.x +  threadIdx.x;
            unsigned int y = blockIdx.y * blockDim.y +  threadIdx.y;
            const unsigned int i = x + y*nbCol;
            unsigned int N; /* Nb cell alive*/
            if(x > 0 && y > 0 && x < nbCol-1 && y < nbRow-1){
                N = ( data[x-1 + (y-1)*nbCol] + data[x + (y-1)*nbCol] + data[x+1 + (y-1)*nbCol] +
                      data[x-1 + ( y )*nbCol] +                       + data[x+1 + ( y )*nbCol] +
                      data[x-1 + (y+1)*nbCol] + data[x + (y+1)*nbCol] + data[x+1 + (y+1)*nbCol] );
                      
                dataRes[i] = ( data[i] & (((N==2) | (N==3)))) /*survive*/
                            | (!data[i] & N==3);               /*birth*/
            }else if((x == 0) || (y == 0) || (x == (nbCol-1)) || (y = (nbRow-1))){
                 dataRes[i] = false;
            }
        }
        """)

        self.cuda_func = mod.get_function("bit_gol")
    
    def _step(self,X,Y):
        self.cuda_func(
                X,Y ,np.uint32(self.shape[0]),np.uint32(self.shape[1]),
                block=self._block, grid=self._grid)

    def run(self,nbIteration=1):
        for i in range(nbIteration):
            X = self._cuda_buffers[int(self._pointer)]
            Y = self._cuda_buffers[int(not(self._pointer))]
            self._step(X,Y)
            self._pointer = not(self._pointer)



if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    size = int(sys.argv[1])
    nbIt = int(sys.argv[2])
    gof = GameOfLifeGPU((size,size))
    gof.set(np.random.random(gof.shape) < 0.5)
    gof.run(nbIt)
    print(gof.get())
    plt.imshow(gof.get(),cmap='gray_r')
    plt.show()


