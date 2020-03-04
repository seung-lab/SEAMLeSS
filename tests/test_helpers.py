import unittest
import torch
import numpy as np
from helpers import *

class TestTensorApproxEq(unittest.TestCase):

  def test_approx_eq(self):
    W = torch.zeros((1,2,2,2))
    self.assertTrue(tensor_approx_eq(W, W+1e-9))


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
    print('test_identity')
    U = torch.zeros((1,2,2,2))
    V = invert(U)
    # V = V.permute((0,3,1,2))
    # UofV = grid_sample(V, U, padding_mode='border')
    # UofV = UofV.permute((0,2,3,1))
    eq = tensor_approx_eq(U, V)
    self.assertTrue(eq)

  def test_shift_down_right(self):
    print('test_shift_down_right')
    U = torch.ones((1,2,2,2))
    V = invert(U)
    _V = -torch.ones((1,2,2,2))
    eq = tensor_approx_eq(V, _V)
    self.assertTrue(eq)
    
  def test_rotate_clockwise(self):
    print('test_rotate_clockwise')
    U = torch.zeros((1,2,2,2))
    U[0,0,0,0] = 1
    U[0,0,1,1] = 1
    U[0,1,1,0] = -1
    U[0,1,0,1] = -1
    V = invert(U)
    _V = torch.zeros((1,2,2,2))
    _V[0,0,0,1] = 1
    _V[0,0,1,0] = -1
    _V[0,1,1,1] = -1
    _V[0,1,0,0] = 1
    eq = tensor_approx_eq(V, _V, 1e-4)
    self.assertTrue(eq)
    W = compose(U, V) 
    eq = tensor_approx_eq(W, torch.zeros_like(W), 1e-4)

  def test_rotate_clockwise3x3(self):
    print('test_rotate_clockwise3x3')
    U = torch.zeros((1,3,3,2))
    U[0,0,0,0] = 2/3.
    U[0,0,1,1] = 2/3. 
    U[0,1,1,0] = -2/3.
    U[0,1,0,1] = -2/3.
    V = invert(U)
    _V = torch.zeros((1,3,3,2))
    _V[0,0,0,1] = 2/3.
    _V[0,0,1,0] = -2/3.
    _V[0,1,1,1] = -2/3.
    _V[0,1,0,0] = 2/3.
    eq = tensor_approx_eq(V, _V, 1e-4)
    self.assertTrue(eq)
    W = compose(U, V) 
    eq = tensor_approx_eq(W, torch.zeros_like(W), 1e-4)

