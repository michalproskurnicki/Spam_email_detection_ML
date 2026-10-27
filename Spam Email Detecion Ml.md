# 📧 SMS Spam Detection (NLP)

Projekt klasyfikacji wiadomości tekstowych jako **SPAM** lub **HAM** (pożądane) przy użyciu przetwarzania języka naturalnego. Model bazuje na wektoryzacji **TF-IDF** oraz algorytmie **Logistic Regression** z równoważeniem klas, co zapewnia wysoką skuteczność na mocno niezbalansowanym zbiorze danych.

## 📊 Wyniki Modelu (Zbiór testowy)
Model został przetestowany z rygorystycznym zachowaniem braku wycieku danych (preprocessing i wektoryzacja aplikowane dopiero po podziale `train_test_split`).

* **ROC-AUC:** `0.9857` – niemal idealna zdolność modelu do rozdzielania klas.
* **Accuracy:** `98.21%` – ogólna poprawność klasyfikacji.
* **Precision (Spam):** `95.10%` – zminimalizowane ryzyko oznaczania ważnych wiadomości jako spam (False Positives).
* **Recall (Spam):** `91.28%` – bardzo wysoka skuteczność wyłapywania faktycznego spamu.
* **F1-Score (Spam):** `0.9315` – świetny balans między precyzją a czułością.

### 📋 Raport Klasyfikacji

                  precision    recall  f1-score   support

             Ham       0.99      0.99      0.99       966
            Spam       0.95      0.91      0.93       149

        accuracy                           0.98      1115
       macro avg       0.97      0.95      0.96      1115
    weighted avg       0.98      0.98      0.98      1115

## 🛠️ Technologie
* **Język:** Python
* **Machine Learning & NLP:** Scikit-learn (LogisticRegression, TfidfVectorizer, train_test_split z opcją stratify)
* **Analiza i Wizualizacja:** Pandas, NumPy, Matplotlib, Seaborn
  
## Wizualizacje
<img width="1385" height="490" alt="Zrzut ekranu 2026-10-27 102858" src="https://github.com/user-attachments/assets/8188819c-41ed-4dac-8674-624ef2a5fd5e" />

<img width="591" height="493" alt="Zrzut ekranu 2026-10-27 102714" src="https://github.com/user-attachments/assets/e20c62a8-147a-4150-bd83-139721ccaaff" />

<img width="796" height="403" alt="Zrzut ekranu 2026-10-27 102703" src="https://github.com/user-attachments/assets/517a657d-ed88-4696-a7d7-99196cd2658d" />





## Autor Michal Proskurnicki- Student 2 roku informatyki i ekonometrii na UEP.
