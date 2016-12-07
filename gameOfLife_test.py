import unittest
import numpy as np
import numpy.testing as npt
from gameOfLifeBool import GameOfLifeBool


class GameOfLifeTest(unittest.TestCase):
    def __init__(self,methodName='test_run',gofClass=GameOfLifeBool):
        super().__init__(methodName)
        self.gofClass = gofClass
        
    def setUp(self):
        self.shape = (6,6)
        self.uut = self.gofClass(self.shape)
        
    def test_run_it(self):
        a = np.array([
                [0,0,0,0,0,0],
                [0,0,0,0,0,0],
                [0,0,1,1,1,0],
                [0,1,1,1,0,0],
                [0,0,0,0,0,0],
                [0,0,0,0,0,0]
            ]).astype(self.uut.get().dtype)
        self.uut.set(a) #initialisation state
        b = np.array([
                [0,0,0,0,0,0],
                [0,0,0,1,0,0],
                [0,1,0,0,1,0],
                [0,1,0,0,1,0],
                [0,0,1,0,0,0],
                [0,0,0,0,0,0]
            ]).astype(self.uut.get().dtype)
        
        self.uut.run(nbIteration=3)
        npt.assert_equal(self.uut.get(),b)
        
    def test_run(self):
        a = np.array([
                [0,0,0,0,0,0],
                [0,0,0,0,0,0],
                [0,0,1,1,1,0],
                [0,1,1,1,0,0],
                [0,0,0,0,0,0],
                [0,0,0,0,0,0]
            ]).astype(self.uut.get().dtype)
        self.uut.set(a) #initialisation state
        b = np.array([
                [0,0,0,0,0,0],
                [0,0,0,1,0,0],
                [0,1,0,0,1,0],
                [0,1,0,0,1,0],
                [0,0,1,0,0,0],
                [0,0,0,0,0,0]
            ]).astype(self.uut.get().dtype)
        
        self.uut.run()
        npt.assert_equal(self.uut.get(),b)
        self.uut.run()
        npt.assert_equal(self.uut.get(),a)
        self.uut.run()
        npt.assert_equal(self.uut.get(),b)

def runGOFTest(gofClass):
        suite = unittest.TestSuite()
        suite.addTest(GameOfLifeTest(gofClass=gofClass))
        suite.addTest(GameOfLifeTest(methodName='test_run_it',gofClass=gofClass))
        unittest.TextTestRunner().run(suite)

if __name__ == "__main__":
    runGOFTest(GameOfLifeBool)
