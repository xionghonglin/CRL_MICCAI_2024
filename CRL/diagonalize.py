import torch


def loss_FO(m):
  b, c, n = m.shape
  m = torch.nn.functional.normalize(m, dim=2, p=2)
  m_T = torch.transpose(m, 1, 2)
  m_cc = torch.matmul(m, m_T)
  mask = torch.eye(c).unsqueeze(0).repeat(b,1,1)
  m_cc = m_cc.masked_fill(mask==1, 0)
  loss = torch.sum(m_cc**2)/(b*c*(c-1))
  return loss

