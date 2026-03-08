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

> <img width="1920" height="1080" alt="Zrzut ekranu (7)" src="https://github.com/user-attachments/assets/cb7e278d-a625-4a15-9812-f3cffe7eefb3" />
<img width="1920" height="1080" alt="Zrzut ekranu (8)" src="https://github.com/user-attachments/assets/4dd1b2c9-c2fd-46cb-bad9-70e2ec7dffbb" />
<img width="1920" height="1080" alt="Zrzut ekranu (9)" src="https://github.com/user-attachments/assets/e808e5f8-1bc6-48bb-bb23-595597cadd19" />
<img width="1920" height="1080" alt="Zrzut ekranu (10)" src="https://github.com/user-attachments/assets/c9641b85-8963-4201-9e48-e12f85d11745" />





---

## 🌟 Future Improvements
- [ ] Add message length as an additional feature to improve model accuracy
- [ ] Try other machine learning models (Random Forest, Naive Bayes, etc.)
- [ ] Create an interactive interface using Streamlit or similar
- [ ] Model Accuracy 0.9497757847533632

---

## 👤 Author
**Michał Proskurnicki** – [GitHub Profile](https://github.com/michalproskurnicki)  
*2nd-year student, Informatics and Econometrics, Poznań University of Economics*
