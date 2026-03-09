# 📧 Spam Email Detection ML

This project classifies Email messages as **spam** or **ham (not spam)**.  
It uses Python, Pandas, scikit-learn, and TF-IDF for text processing, and Logistic Regression for prediction.

---

## ⚙️ Features
- Load and clean SMS dataset (`spam.csv`)
- Perform statistical analysis (number of messages, average text length)
- Visualize data (average message length, most frequent words)
- SMS classification using Logistic Regression
- Confusion matrix and classification report
- Test the model on new, unseen messages
- Model Accuracy 0.9497757847533632

---

## 🛠️ Technologies
- **Python 3.14.3**
- Pandas
- Numpy
- Matplotlib / Seaborn
- scikit-learn (TF-IDF, Logistic Regression)

---

## 🚀 Installation & Usage

**1. Clone the repository:**
```bash
git clone [https://github.com/michalproskurnicki/Spam_email_detection_ML.git](https://github.com/michalproskurnicki/Spam_email_detection_ML.git)
```

**2. Install required packages:**
```bash
pip install -r requirements.txt
```

**3. Dataset:**
Place `spam.csv` in the project folder or update the path in the code.

**4. Run the project:**
```bash
python spam_detector.py
```

---

## 💻 Example Usage

```python
my_text = "Congratulations! You've won a free ticket to the Bahamas. Click here to claim your prize."
prediction = model.predict(vectorizer.transform([my_text]))
print(prediction) # 1 = spam, 0 = ham
```

---

## 📊 Visualizations
- Average message length by type (spam vs ham)
- Confusion matrix
- Most indicative words for spam and ham messages

> <img width="1643" height="924" alt="Zrzut ekranu (15)" src="https://github.com/user-attachments/assets/5b37d8d2-8f89-4ea7-9b74-6687250ce67c" />
><img width="1641" height="923" alt="Zrzut ekranu (16)" src="https://github.com/user-attachments/assets/562d4b17-5059-42e2-b9bb-d984c0457098" />
><img width="1653" height="930" alt="Zrzut ekranu (17)" src="https://github.com/user-attachments/assets/11414d61-0d61-4d02-bd4e-9925156bf60e" />
><img width="1648" height="927" alt="Zrzut ekranu (18)" src="https://github.com/user-attachments/assets/8d314d26-1b57-4c00-91b7-400b7022aad2" />








---

## 🌟 Future Improvements
- [ ] Add message length as an additional feature to improve model accuracy
- [ ] Try other machine learning models (Random Forest, Naive Bayes, etc.)
- [ ] Create an interactive interface using Streamlit or similar
      

---

## 👤 Author
**Michał Proskurnicki** – [GitHub Profile](https://github.com/michalproskurnicki)  
*2nd-year student, Informatics and Econometrics, Poznań University of Economics*
