import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# -----------------------
# VERİ YÜKLE & MODEL EĞİT
# -----------------------
df = pd.read_csv("train.csv")

df = df.drop(["Cabin", "Name", "Ticket", "PassengerId"], axis=1)

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

X = df.drop("Survived", axis=1)
y = df["Survived"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -----------------------
# UI
# -----------------------
st.title("🚢 Titanic Survival Prediction")

pclass = st.selectbox("Sınıf", [1, 2, 3])
sex = st.selectbox("Cinsiyet", ["Erkek", "Kadın"])
age = st.slider("Yaş", 1, 80, 25)
sibsp = st.number_input("Kardeş/Eş sayısı", 0, 10, 0)
parch = st.number_input("Ebeveyn/Çocuk sayısı", 0, 10, 0)
fare = st.number_input("Bilet Ücreti", 0.0, 500.0, 10.0)

embarked = st.selectbox("Limana Biniş", ["S", "C", "Q"])

# -----------------------
# INPUT HAZIRLA
# -----------------------
sex = 1 if sex == "Kadın" else 0

embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0

input_data = pd.DataFrame([[
    pclass, sex, age, sibsp, parch, fare, embarked_Q, embarked_C
]], columns=X.columns)

# -----------------------
# TAHMİN
# -----------------------
if st.button("Tahmin Et"):
    result = model.predict(input_data)

    if result[0] == 1:
        st.success("👉 Hayatta kalır")
    else:
        st.error("👉 Hayatta kalamaz")