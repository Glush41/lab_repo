import pandas as pd

lab2 = pd.read_csv("ml_metrics.csv")
lab3 = pd.read_csv("dl_metrics.csv")

lab2_acc = lab2[lab2["mode"] == "full"]["accuracy"].values[0]
lab3_acc = lab3[lab3["mode"] == "full"]["best_val_acc"].values[0]

print(f"ЛР2 (HOG+SGD) full accuracy: {lab2_acc:.4f}")
print(f"ЛР3 (CNN) full best_val_acc: {lab3_acc:.4f}")
print(f"Прирост CNN над ML: {lab3_acc / lab2_acc:.2f}x")
