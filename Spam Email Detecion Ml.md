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

> *Tip: You can add images of these plots here for better visual impact.*

---

## 🌟 Future Improvements
- [ ] Add message length as an additional feature to improve model accuracy
- [ ] Try other machine learning models (Random Forest, Naive Bayes, etc.)
- [ ] Create an interactive interface using Streamlit or similar

---

## 👤 Author
**Michał Proskurnicki** – [GitHub Profile](https://github.com/michalproskurnicki)  
*2nd-year student, Informatics and Econometrics, Poznań University of Economics*
