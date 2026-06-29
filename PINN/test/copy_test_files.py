import os
import shutil

scratch_dir = r"C:\Users\yeonj\.gemini\antigravity-ide\scratch"
target_dir = r"C:\Users\yeonj\OneDrive\바탕 화면\DeepLearning_code\PINN\test"

os.makedirs(target_dir, exist_ok=True)

files_to_copy = [
    "inspect_notebook.py",
    "fix_notebook.py",
    "run_pinn_check.py",
    "run_pinn_lbfgs.py",
    "debug_losses.py",
    "test_pinn_supervised.py",
    "test_autograd.py",
    "check_supervised_pde.py",
    "run_pinn_high_weight.py",
    "find_cell.py",
    "update_notebook.py"
]

copied_files = []
for file_name in os.listdir(scratch_dir):
    if file_name in files_to_copy:
        src_path = os.path.join(scratch_dir, file_name)
        dst_path = os.path.join(target_dir, file_name)
        shutil.copy2(src_path, dst_path)
        copied_files.append(file_name)

print("Copied files:")
for f in copied_files:
    print(f" - {f}")
print(f"Successfully copied to: {target_dir}")
