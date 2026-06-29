import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

N_u = 600
N_f = 2000

# Positive pole: center (0, 0.35), radius 0.15, V = 1.0
# Negative pole: center (0, -0.35), radius 0.15, V = -1.0

# 1. Generate boundary and interior points for positive pole
# We sample points uniformly inside and on the boundary of the positive circle
r_p = np.sqrt(np.random.uniform(0, 0.15**2, (200, 1)))
theta_p = np.random.uniform(0, 2*np.pi, (200, 1))
x_pos = r_p * np.cos(theta_p)
y_pos = 0.35 + r_p * np.sin(theta_p)
V_pos = np.ones_like(x_pos) * 1.0
X_pos = np.hstack((x_pos, y_pos))

# 2. Generate boundary and interior points for negative pole
r_n = np.sqrt(np.random.uniform(0, 0.15**2, (200, 1)))
theta_n = np.random.uniform(0, 2*np.pi, (200, 1))
x_neg = r_n * np.cos(theta_n)
y_neg = -0.35 + r_n * np.sin(theta_n)
V_neg = np.ones_like(x_neg) * -1.0
X_neg = np.hstack((x_neg, y_neg))

# 3. Generate boundary points for the outer boundary (V = 0)
x_l = -np.ones((50, 1))
y_l = np.random.uniform(-1.0, 1.0, (50, 1))
x_r = np.ones((50, 1))
y_r = np.random.uniform(-1.0, 1.0, (50, 1))
y_b = -np.ones((50, 1))
x_b = np.random.uniform(-1.0, 1.0, (50, 1))
y_t = np.ones((50, 1))
x_t = np.random.uniform(-1.0, 1.0, (50, 1))
X_outer = np.vstack((np.hstack((x_l, y_l)), np.hstack((x_r, y_r)), np.hstack((x_b, y_b)), np.hstack((x_t, y_t))))
V_outer = np.zeros((X_outer.shape[0], 1))

X_u = torch.FloatTensor(np.vstack((X_pos, X_neg, X_outer)))
V_u = torch.FloatTensor(np.vstack((V_pos, V_neg, V_outer)))

# Generate free space points (exclude the pole regions)
x_f = np.random.uniform(-1.0, 1.0, (N_f * 3, 1))
y_f = np.random.uniform(-1.0, 1.0, (N_f * 3, 1))
X_f_np = np.hstack((x_f, y_f))

dist_pos = np.sqrt(X_f_np[:, 0]**2 + (X_f_np[:, 1] - 0.35)**2)
dist_neg = np.sqrt(X_f_np[:, 0]**2 + (X_f_np[:, 1] + 0.35)**2)
mask = (dist_pos > 0.15) & (dist_neg > 0.15)
X_f_np = X_f_np[mask][:N_f]

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

hidden_layers = [32, 32, 32]
model = DynamicTanhPINN(input_size=2, hidden_layers=hidden_layers, output_size=1)
optimizer = optim.Adam(model.parameters(), lr=2e-3) # slightly higher lr

print("Starting dipole training using standard Adam...")
for epoch in range(1500):
    optimizer.zero_grad()
    u_pred_bc, _ = model(X_u)
    Loss_BC = nn.MSELoss()(u_pred_bc, V_u)
    Loss_PDE = calc_pde_loss(model, X_f)
    loss = 50.0 * Loss_BC + Loss_PDE # BC weight is 50.0, very standard
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 300 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f} (BC: {Loss_BC.item():.4f}, PDE: {Loss_PDE.item():.4f})")

# Verify predictions
print("\n--- Dipole predictions at key points ---")
with torch.no_grad():
    print(f"V(0.0, 0.35) [Positive Center, Target=1.0]: {model(torch.FloatTensor([[0.0, 0.35]]))[0].item():.4f}")
    print(f"V(0.0, -0.35) [Negative Center, Target=-1.0]: {model(torch.FloatTensor([[0.0, -0.35]]))[0].item():.4f}")
    print(f"V(0.0, 0.0) [Symmetry Line, Target=0.0]: {model(torch.FloatTensor([[0.0, 0.0]]))[0].item():.4f}")
    print(f"V(0.0, 0.6) [Above Positive]: {model(torch.FloatTensor([[0.0, 0.6]]))[0].item():.4f}")
