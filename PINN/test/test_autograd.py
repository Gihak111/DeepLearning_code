import torch

x = torch.tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)

# V = x^3 + y^3
# for row 0: V = 2^3 + 3^3 = 8 + 27 = 35. x_0 = 2, y_0 = 3.
# dV_dx = 3x^2 = 3*(4) = 12. dV_dy = 3y^2 = 3*(9) = 27.
# d2V_dx2 = 6x = 12. d2V_dy2 = 6y = 18.
# for row 1: V = 4^3 + 5^3 = 64 + 125 = 189. x_1 = 4, y_1 = 5.
# dV_dx = 3*(16) = 48. dV_dy = 3*(25) = 75.
# d2V_dx2 = 6*(4) = 24. d2V_dy2 = 6*(5) = 30.

V = x[:, 0:1]**3 + x[:, 1:2]**3

grad_V = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0]
print("grad_V:")
print(grad_V)

dV_dx = grad_V[:, 0:1]
grad_dV_dx = torch.autograd.grad(dV_dx, x, grad_outputs=torch.ones_like(dV_dx), create_graph=True)[0]
d2V_dx2 = grad_dV_dx[:, 0:1]

dV_dy = grad_V[:, 1:2]
grad_dV_dy = torch.autograd.grad(dV_dy, x, grad_outputs=torch.ones_like(dV_dy), create_graph=True)[0]
d2V_dy2 = grad_dV_dy[:, 1:2]

print("d2V_dx2 (expected [[12], [24]]):")
print(d2V_dx2)
print("d2V_dy2 (expected [[18], [30]]):")
print(d2V_dy2)
