import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

N_u = 600
N_f = 2000

x_g = np.random.uniform(-1.0, 1.0, (N_u // 3, 1))
y_g = np.zeros_like(x_g)
V_g = np.zeros_like(x_g)
X_g = np.hstack((x_g, y_g))

x_p = np.random.uniform(-0.4, 0.4, (N_u // 3, 1))
y_p = np.ones_like(x_p) * 0.2
V_p = np.ones_like(x_p) * 1.0
X_p = np.hstack((x_p, y_p))

x_l = -np.ones((N_u // 9, 1))
y_l = np.random.uniform(0.0, 1.0, (N_u // 9, 1))
x_r = np.ones((N_u // 9, 1))
y_r = np.random.uniform(0.0, 1.0, (N_u // 9, 1))
x_t = np.random.uniform(-1.0, 1.0, (N_u // 9, 1))
y_t = np.ones_like(x_t)
X_far = np.vstack((np.hstack((x_l, y_l)), np.hstack((x_r, y_r)), np.hstack((x_t, y_t))))
V_far = np.zeros((X_far.shape[0], 1))

X_u = torch.FloatTensor(np.vstack((X_g, X_p, X_far)))
V_u = torch.FloatTensor(np.vstack((V_g, V_p, V_far)))

x_f = np.random.uniform(-1.0, 1.0, (N_f, 1))
y_f = np.random.uniform(0.0, 1.0, (N_f, 1))
X_f_np = np.hstack((x_f, y_f))
mask_patch = (np.abs(X_f_np[:, 0]) <= 0.4) & (np.abs(X_f_np[:, 1] - 0.2) < 0.01)
X_f_np = X_f_np[~mask_patch]
X_f = torch.FloatTensor(X_f_np)
X_f.requires_grad_(True)

class TanhLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(TanhLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        return torch.tanh(self.linear(x))

class DynamicTanhPINN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(DynamicTanhPINN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(TanhLayer(input_size, hidden_layers[0]))
        for i in range(len(hidden_layers)-1):
            self.layers.append(TanhLayer(hidden_layers[i], hidden_layers[i+1]))
        self.output_layer = nn.Linear(hidden_layers[-1], output_size)
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.output_layer(out)
        return out, None

def calc_pde_loss(model_net, x_points):
    pred, _ = model_net(x_points)
    grad_V = torch.autograd.grad(pred, x_points, grad_outputs=torch.ones_like(pred), create_graph=True)[0]
    dV_dx = grad_V[:, 0:1]
    dV_dy = grad_V[:, 1:2]
    grad_dV_dx = torch.autograd.grad(dV_dx, x_points, grad_outputs=torch.ones_like(dV_dx), create_graph=True)[0]
    d2V_dx2 = grad_dV_dx[:, 0:1]
    grad_dV_dy = torch.autograd.grad(dV_dy, x_points, grad_outputs=torch.ones_like(dV_dy), create_graph=True)[0]
    d2V_dy2 = grad_dV_dy[:, 1:2]
    return torch.mean((d2V_dx2 + d2V_dy2)**2)

hidden_layers = [64, 64, 64, 64, 64]
model = DynamicTanhPINN(input_size=2, hidden_layers=hidden_layers, output_size=1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1000):
    optimizer.zero_grad()
    
    # Evaluate predictions on parts
    u_pred_g, _ = model(torch.FloatTensor(X_g))
    u_pred_p, _ = model(torch.FloatTensor(X_p))
    u_pred_far, _ = model(torch.FloatTensor(X_far))
    
    Loss_BC_g = nn.MSELoss()(u_pred_g, torch.FloatTensor(V_g))
    Loss_BC_p = nn.MSELoss()(u_pred_p, torch.FloatTensor(V_p))
    Loss_BC_far = nn.MSELoss()(u_pred_far, torch.FloatTensor(V_far))
    
    Loss_BC = Loss_BC_g + Loss_BC_p + Loss_BC_far
    Loss_PDE = calc_pde_loss(model, X_f)
    
    loss = 50.0 * Loss_BC + Loss_PDE
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}:")
        print(f"  Loss_BC_g: {Loss_BC_g.item():.4f}, Loss_BC_p: {Loss_BC_p.item():.4f}, Loss_BC_far: {Loss_BC_far.item():.4f}")
        print(f"  Loss_PDE: {Loss_PDE.item():.4f}, Total Loss: {loss.item():.4f}")
