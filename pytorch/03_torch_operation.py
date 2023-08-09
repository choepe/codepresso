import torch

# shape=(2, 3) torch.tensor
tensor_a = torch.Tensor([[1, 2, 3],
                         [4, 5, 6]])

tensor_b = torch.Tensor([[7, 8, 9],
                         [10, 11, 12]])

# add
tensor_a + tensor_b
# mul
tensor_a * tensor_b
# dot
tensor_b.T.shape
torch.matmul(input=tensor_a, other=tensor_b.T)
# sqr
torch.square(tensor_a)
# log
torch.log(tensor_a)
# gradient
tensor_a.requires_grad, tensor_b.requires_grad
tensor_a = torch.tensor([[1., 2., 3.],
                         [4., 5., 6.]], requires_grad=True)

tensor_b = torch.tensor([[7., 8., 9.],
                         [10., 11., 12.]], requires_grad=True)
tensor_a.requires_grad, tensor_b.requires_grad
print('tensor_a의 미분값: ', tensor_a.grad)
print('tensor_b의 미분값: ', tensor_b.grad)

tensor_c = torch.square(tensor_a + tensor_b)
result = torch.sum(tensor_c)
result.backward()
print('tensor_a의 미분값: ', tensor_a.grad)
print('tensor_b의 미분값: ', tensor_b.grad)

# shape=(2, 3), value=1
external_grad = torch.ones(size=(2, 3))
tensor_c.backward(gradient=external_grad)
print('tensor_a의 미분값: ', tensor_a.grad)
print('tensor_b의 미분값: ', tensor_b.grad)