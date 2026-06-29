import json

path = r"C:\Users\yeonj\OneDrive\바탕 화면\DeepLearning_code\PINN\reflect.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source_text = "".join(cell["source"])
        if "model_final = DynamicTanhPINN" in source_text:
            print(f"Found target code cell at index {idx}:")
            print("".join(cell["source"][:15]))
            print("...")
            break
else:
    print("Target cell not found!")
