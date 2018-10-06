import cpc
import torch
import unittest

class TestCPC(unittest.TestCase):

  def setUp(self):
    n = 16
    a = np.array([[i%2. for i in range(n)] for j in range(n)])
    b = np.array([[(i-1)%2. for i in range(n)] for j in range(n)])
    S = torch.from_numpy(a)
    T = torch.from_numpy(b)
    self.S = S.unsqueeze(0).unsqueeze(0)
    self.T = T.unsqueeze(0).unsqueeze(0)
    return self.S, self.T

  def test_cpc(self):
    R_hat = cpc.cpc(self.S, self.T, 4)
    R = -torch.ones([1,1,4,4], dtype=torch.float64)
    self.assertEqual(R_hat, R)
