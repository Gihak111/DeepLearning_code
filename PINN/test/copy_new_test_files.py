import os
import shutil

scratch_dir = r"C:\Users\yeonj\.gemini\antigravity-ide\scratch"
target_dir = r"C:\Users\yeonj\OneDrive\바탕 화면\DeepLearning_code\PINN\test"

files_to_copy = [
    "update_notebook_lbfgs.py",
    "run_pinn_pretrain_lbfgs.py"
]

copied_files = []
for file_name in files_to_copy:
    src_path = os.path.join(scratch_dir, file_name)
    dst_path = os.path.join(target_dir, file_name)
    shutil.copy2(src_path, dst_path)
    copied_files.append(file_name)

print("Copied files:")
for f in copied_files:
    print(f" - {f}")
