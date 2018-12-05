import unittest
import torch
import numpy as np
from helpers import *

def tensor_approx_eq(A, B, eta=1e-7):
  return torch.all(torch.lt(torch.abs(torch.add(A, -B)), 1e-7)) 

class TestRelToGrid(unittest.TestCase):
  
  def test_identity(self):
    U = identity_grid(3)
    V = rel_to_grid(U)
    _V = torch.zeros_like(U)
    for i in range(3):
      for j in range(3):
        _V[0,i,j,0] = j
        _V[0,i,j,1] = i 
    eq = tensor_approx_eq(V, _V)
    self.assertTrue(eq)

class TestGridToRel(unittest.TestCase):
  
  def test_identity(self):
    _V = identity_grid(3)
    U = torch.zeros_like(_V)
    for i in range(3):
      for j in range(3):
        U[0,i,j,0] = j
        U[0,i,j,1] = i 
    V = grid_to_rel(U)
    eq = tensor_approx_eq(V, _V)
    self.assertTrue(eq)

class TestInvert(unittest.TestCase):

  def test_identity(self):
    U = torch.zeros((1,2,2,2))
    V = invert(U)
    # V = V.permute((0,3,1,2))
    # UofV = gridsample_residual(V, U, padding_mode='border')
    # UofV = UofV.permute((0,2,3,1))
    eq = tensor_approx_eq(U, V)
    self.assertTrue(eq)

  def test_shift_right(self):
    U = torch.zeros((1,2,2,2))
    U[0,0,0,0] = 1 # vector from [0,0] to [1,0]
    U[0,1,0,0] = 1 # vector from [1,0] to [2,0] out of image
    V = invert(U)
    _V = torch.zeros((1,2,2,2))
    _V[0,1,0,0] = -1
    _V[0,0,0,0] = -10 
    _V[0,0,0,1] = -10 
    V[torch.isnan(V)] = -10
    eq = tensor_approx_eq(V, _V)
    self.assertTrue(eq)
    
  def test_rotate_clockwise(self):
    U = torch.zeros((1,3,3,2))
    U[0,0,0,0] = 1
    U[0,1,0,1] = 1
    U[0,1,1,0] = -1
    U[0,0,1,1] = -1
    V = invert(U)
    _V = torch.zeros((1,3,3,2))
    _V[0,0,0,1] = 1
    _V[0,1,0,0] = -1
    _V[0,1,1,1] = -1
    _V[0,0,1,0] = 1
    eq = tensor_approx_eq(V, _V)
    self.assertTrue(eq)
    W = U + gridsample_residual(V.permute(0,3,1,2), U, 'border').permute(0,2,3,1)
    eq = tensor_approx_eq(W, torch.zeros(W))

  def test_multi_src_vector(self):
    U = torch.zeros((1,2,2,2))
    U[0,0,0,0] = 1 # vector from [0,0] to [1,0]
    V = invert(U)
    _V = torch.zeros((1,2,2,2))
    _V[0,1,0,0] = -0.5
    _V[0,0,0,0] = -10 
    _V[0,0,0,1] = -10 
    V[torch.isnan(V)] = -10
    eq = tensor_approx_eq(V, _V)
    self.assertTrue(eq)
