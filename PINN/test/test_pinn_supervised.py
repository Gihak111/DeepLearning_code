import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

N_u = 600

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

hidden_layers = [64, 64, 64, 64, 64]
model = DynamicTanhPINN(input_size=2, hidden_layers=hidden_layers, output_size=1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Training ONLY on BC loss...")
for epoch in range(2000):
    optimizer.zero_grad()
    u_pred_bc, _ = model(X_u)
    loss = nn.MSELoss()(u_pred_bc, V_u)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 400 == 0:
        print(f"Epoch {epoch+1}, BC Loss: {loss.item():.6f}")

with torch.no_grad():
    print("\n--- Predictions after supervised training ---")
    print(f"V(0.0, 0.0) [Ground, Target=0.0]: {model(torch.FloatTensor([[0.0, 0.0]]))[0].item():.4f}")
    print(f"V(0.0, 0.2) [Patch, Target=1.0]: {model(torch.FloatTensor([[0.0, 0.2]]))[0].item():.4f}")
    print(f"V(0.0, 0.5) [Above patch]: {model(torch.FloatTensor([[0.0, 0.5]]))[0].item():.4f}")
