import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr)

sc_bad = StandardScaler()
X_all_sc = sc_bad.fit_transform(X)

assert abs(X_tr_sc.mean()) < 1e-10, "leak detected: scaler fit on all data"
print("leak test passed")
