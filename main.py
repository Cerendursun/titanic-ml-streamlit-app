import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. VERİYİ YÜKLE
df = pd.read_csv("train.csv")

# 2. GEREKSİZ SÜTUNLARI SİL (ÖNEMLİ: PassengerId de siliniyor)
df = df.drop(["Cabin", "Name", "Ticket", "PassengerId"], axis=1)

# 3. EKSİK VERİLERİ DOLDUR
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# 4. KATEGORİK VERİLERİ SAYISALA ÇEVİR
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# 5. X ve y
X = df.drop("Survived", axis=1)
y = df["Survived"]

# 6. TRAIN / TEST AYIR
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. MODEL
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. TAHMİN + ACCURACY
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


print("\n--- Yeni Yolcu ---")


new_passenger = pd.DataFrame(
    [[3, 0, 25, 0, 0, 7.25, 0, 1]],
    columns=X.columns
)

result = model.predict(new_passenger)

if result[0] == 1:
    print("👉 Hayatta kalır")
else:
    print("👉 Hayatta kalamaz")

