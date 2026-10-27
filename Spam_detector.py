import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    classification_report, 
    confusion_matrix
)

# 1. Wczytanie danych - właściwa ścieżka do pliku .csv
file_path = r"C:\Users\PC\Desktop\spam py\spam.csv"

spam = pd.read_csv(file_path, encoding='latin-1')

# Usunięcie niepotrzebnych kolumn
cols_to_drop = [c for c in ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"] if c in spam.columns]
spam = spam.drop(columns=cols_to_drop)

# Zmiana nazw kolumn na czytelniejsze
spam = spam.rename(columns={"v1": "target", "v2": "text"})

# Mapowanie klas: ham = 0, spam = 1
spam["target"] = spam["target"].map({"ham": 0, "spam": 1}).astype(int)

# 2. Przegląd danych
print("--- Pierwsze 5 wierszy ---")
print(spam.head())
print("\n--- Informacje o zbiorze ---")
print(spam.info())

# Rozkład klas 
print("\n--- Rozkład klas ---")
print(spam["target"].value_counts(normalize=True))

# Podstawowy wykres długości wiadomości 
spam['char_length'] = spam['text'].str.len()
plt.figure(figsize=(8, 4))
sns.barplot(
    x="target", 
    y="char_length", 
    hue="target", 
    data=spam, 
    errorbar=None, 
    palette=["blue", "red"], 
    legend=False
)
plt.xticks([0, 1], ["Ham", "Spam"])
plt.xlabel("Typ wiadomości")
plt.ylabel("Średnia liczba znaków")
plt.title("Średnia długość wiadomości według typu")
plt.show()

# 3. Funkcja czyszcząca tekst
def clean_text(series: pd.Series) -> pd.Series:
    series = series.astype(str)
    series = series.str.replace(r"[^\w\s]", "", regex=True)  
    series = series.str.replace(r"\s+", " ", regex=True)     
    series = series.str.strip()                             
    series = series.str.lower()                             
    return series

# 4. Podział na zbiór treningowy i testowy 
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    spam["text"], 
    spam["target"], 
    test_size=0.2, 
    random_state=42, 
    stratify=spam["target"]
)


X_train = clean_text(X_train_raw)
X_test = clean_text(X_test_raw)

# 5. Wektorowanie TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Trening modelu
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_vec, y_train)

# 7. Predykcja i rozszerzona ewaluacja metryk
y_pred = model.predict(X_test_vec)
y_pred_proba = model.predict_proba(X_test_vec)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*40)
print("--- SZCZEGÓŁOWE METRYKI MODELU ---")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f} (jak często gdy mówimy SPAM, to naprawdę jest SPAM)")
print(f"Recall:    {rec:.4f} (ile całego SPAMU udało się wyłapać)")
print(f"F1-Score:  {f1:.4f} (średnia harmoniczna Precision i Recall)")
print(f"ROC-AUC:   {roc_auc:.4f} (jakość rozdzielania klas przez model)")
print("="*40 + "\n")

print("--- Raport Klasyfikacji ---")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# Macierz pomyłek (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel('Predykcja')
plt.ylabel('Rzeczywistość')
plt.title('Macierz Pomyłek (Confusion Matrix)')
plt.show()

# 8. Test na nowej wiadomości
new_text = "Congratulations! You've won a free ticket to the Bahamas. Click here to claim your prize."
cleaned_new = clean_text(pd.Series([new_text]))
new_vec = vectorizer.transform(cleaned_new)

pred_class = model.predict(new_vec)[0]
pred_prob = model.predict_proba(new_vec)[0][1]

print(f"\nTestowa wiadomość: '{new_text}'")
print(f"Wynik: {'SPAM' if pred_class == 1 else 'HAM'} (Prawdopodobieństwo Spamu: {pred_prob:.2%})")

# 9. Analiza najważniejszych słów
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]
words_weight = pd.DataFrame({'word': feature_names, 'weight': coefficients})

top_spam = words_weight.sort_values(by='weight', ascending=False).head(15)
top_ham = words_weight.sort_values(by='weight', ascending=True).head(15)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].barh(top_spam["word"], top_spam["weight"], color='red')
axes[0].set_title("Top 15 słów wskazujących na SPAM")
axes[0].invert_yaxis()

axes[1].barh(top_ham["word"], top_ham["weight"], color='blue')
axes[1].set_title("Top 15 słów wskazujących na HAM")
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()
