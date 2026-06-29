path = r"C:\Users\yeonj\OneDrive\바탕 화면\DeepLearning_code\PINN\reflect.ipynb"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print("Original lines:")
print("254:", repr(lines[254]))
print("256:", repr(lines[256]))
print("257:", repr(lines[257]))

# Apply fix
lines[254] = '    "    title = f\\"Layer {i+1} Output\\" if i < len(activations)-1 else \\"Final Output (V)\\"\\n",\n'
lines[256] = '    "    ax.set_xlabel(\\"Nodes\\" if i < len(activations)-1 else \\"Channel\\")\\n",\n'
lines[257] = '    "    if i == 0: ax.set_ylabel(\\"Data Index\\")\\n",\n'

print("\nNew lines:")
print("254:", repr(lines[254]))
print("256:", repr(lines[256]))
print("257:", repr(lines[257]))

with open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("\nSuccessfully updated the file!")
