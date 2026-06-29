import json

path = r"C:\Users\yeonj\OneDrive\바탕 화면\DeepLearning_code\PINN\reflect.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update cell index 1 (Domain and Collocation Points)
source_cell_1 = nb["cells"][1]["source"]
updated_source_cell_1 = []
for line in source_cell_1:
    if "mask_patch = " in line and "0.01" in line:
        # replace 0.01 with 0.05 to avoid the derivative singularity near the patch
        new_line = line.replace("0.01", "0.05")
        updated_source_cell_1.append(new_line)
        print("Updated mask_patch line in cell 1!")
    else:
        updated_source_cell_1.append(line)
nb["cells"][1]["source"] = updated_source_cell_1

# Update cell index 13 (Training Loop)
new_training_source = [
    "# ==========================================\n",
    "# 7. 반복 학습 수행 및 결과 시각화 (전위)\n",
    "# ==========================================\n",
    "\n",
    "# 본격적인 학습을 위해 완전히 깨끗한 새 모델을 초기화한다\n",
    "model_final = DynamicTanhPINN(input_size=2, hidden_layers=hidden_layers, output_size=1)\n",
    "\n",
    "loss_history_final = []\n",
    "start_time = time.time()\n",
    "\n",
    "# 7-1. 1단계: 경계 조건(BC) 사전 학습 (Pre-training)\n",
    "# 물리 법칙을 가하기 전에 경계 조건을 먼저 모델에게 가르칩니다.\n",
    "print(\"--- [1단계] 경계 조건 사전 학습 시작 (1000 에포크) ---\")\n",
    "optimizer_pre = optim.Adam(model_final.parameters(), lr=1e-3)\n",
    "for epoch in range(1000):\n",
    "    optimizer_pre.zero_grad()\n",
    "    u_pred_bc, _ = model_final(X_u)\n",
    "    Loss_BC = nn.MSELoss()(u_pred_bc, V_u)\n",
    "    Loss_BC.backward()\n",
    "    optimizer_pre.step()\n",
    "    if (epoch + 1) % 200 == 0:\n",
    "        print(f\"Epoch [{epoch+1:4d}/1000], BC Loss: {Loss_BC.item():.6e}\")\n",
    "\n",
    "# 7-2. 2단계: L-BFGS 옵티마이저를 사용한 물리 법칙(PDE) 결합 학습\n",
    "# 2차 최적화 알고리즘인 L-BFGS를 사용하여 매우 높은 정확도로 물리 법칙을 수렴시킵니다.\n",
    "print(\"\\n--- [2단계] L-BFGS 물리 법칙 결합 학습 시작 ---\")\n",
    "optimizer_final = optim.LBFGS(model_final.parameters(), lr=0.1, max_iter=50, history_size=100)\n",
    "\n",
    "# L-BFGS는 에포크 개념이 일반 Optimizer와 다르므로 40번의 스텝(Step)으로 학습을 진행합니다.\n",
    "for epoch in range(40):\n",
    "    def closure():\n",
    "        optimizer_final.zero_grad()\n",
    "        u_pred_bc, _ = model_final(X_u)\n",
    "        Loss_BC = nn.MSELoss()(u_pred_bc, V_u)\n",
    "        Loss_PDE = calc_pde_loss(model_final, X_f)\n",
    "        loss = 100.0 * Loss_BC + Loss_PDE # BC 가중치와 PDE 균형 결합\n",
    "        loss.backward()\n",
    "        return loss\n",
    "    \n",
    "    optimizer_final.step(closure)\n",
    "    \n",
    "    # 진행도 및 오차 출력\n",
    "    u_pred_bc_val, _ = model_final(X_u)\n",
    "    Loss_BC_val = nn.MSELoss()(u_pred_bc_val, V_u).item()\n",
    "    Loss_PDE_val = calc_pde_loss(model_final, X_f).item()\n",
    "    total_loss_val = 100.0 * Loss_BC_val + Loss_PDE_val\n",
    "    loss_history_final.append(total_loss_val)\n",
    "    \n",
    "    print(f\"Step [{epoch+1:2d}/40], Total Loss: {total_loss_val:.6e} (BC: {Loss_BC_val:.6e}, PDE: {Loss_PDE_val:.6e})\")\n",
    "\n",
    "print(f\"\\n전위 학습 완료! (소요 시간: {time.time() - start_time:.2f}초)\")\n",
    "\n",
    "# 학습 경과 그래프와 전위 히트맵 시각화 패널\n",
    "fig = plt.figure(figsize=(15, 5))\n",
    "\n",
    "# 1. Loss 곡선 시각화\n",
    "ax1 = fig.add_subplot(1, 2, 1)\n",
    "ax1.plot(loss_history_final, color='blue', linewidth=2)\n",
    "ax1.set_yscale('log')\n",
    "ax1.set_title('Training Loss Curve')\n",
    "ax1.set_xlabel('L-BFGS Steps')\n",
    "ax1.set_ylabel('Total PINN Loss')\n",
    "ax1.grid(True, linestyle='--', alpha=0.6)\n",
    "\n",
    "# 2. 예측된 전위 V(x,y) 히트맵\n",
    "ax2 = fig.add_subplot(1, 2, 2)\n",
    "x_grid = np.linspace(-1.0, 1.0, 100)\n",
    "y_grid = np.linspace(0.0, 1.0, 100)\n",
    "X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)\n",
    "grid_tensor = torch.FloatTensor(np.c_[X_mesh.ravel(), Y_mesh.ravel()])\n",
    "\n",
    "with torch.no_grad():\n",
    "    V_pred, _ = model_final(grid_tensor)\n",
    "V_pred = V_pred.numpy().reshape(X_mesh.shape)\n",
    "\n",
    "contour = ax2.contourf(X_mesh, Y_mesh, V_pred, levels=50, cmap='viridis')\n",
    "fig.colorbar(contour, ax=ax2, label='Potential (V)')\n",
    "ax2.set_title(\"Predicted Potential V(x,y)\")\n",
    "ax2.set_xlabel(\"x (m)\")\n",
    "ax2.set_ylabel(\"y (m)\")\n",
    "ax2.plot([-0.4, 0.4], [0.2, 0.2], 'r-', linewidth=4, label='Patch')\n",
    "ax2.plot([-1.0, 1.0], [0.0, 0.0], 'k-', linewidth=4, label='Ground')\n",
    "ax2.legend(loc='upper right')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

nb["cells"][13]["source"] = new_training_source
print("Updated training loop in cell 13!")

# Save back
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Saved reflect.ipynb successfully!")
