import unittest
import torch
from helpers import *

class TestInvert(unittest.TestCase):

  def test_identity(self):
    U = identity_grid(3)
    V = invert(U)
    V = V.permute((0,3,1,2))
    UofV = gridsample_residual(V, U, padding_mode='border')
    UofV = UofV.permute((0,2,3,1))
    self.assertEqual(UofV, U)

  def test_one_vector(self):
    U = identity_grid(3)
    U[0,0,0,0] = 1
    V = invert(U)
    _V = identity_grid(3)
    _V[0,1,0,0] = -1
    self.assertEqual(UofV, U)
    
