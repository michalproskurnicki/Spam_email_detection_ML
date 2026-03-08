# Spam_email_detedion_ML
This project classifies Email messages as spam or ham (not spam).
It uses Python, Pandas, scikit-learn, and TF-IDF for text processing, and Logistic Regression for prediction.

Features:
-Load and clean SMS dataset (spam.csv)
-Statistical analysis of the data (number of messages, average text length)
-Data visualization (average message length, most frequent words)
-SMS classification using Logistic Regression
-Confusion matrix and classification report
-Test the model on new, unseen messages

Technologies
-Python 3.14.3
-Pandas
-Numpy
-Matplotlib / Seaborn
-scikit-learn (TF-IDF, Logistic Regression)

Installation & Usage
1.Clone the repository:
git clone https://github.com/your_username/Spam_email_detection_ML.git

2.Install required packages:
pip install -r requirements.txt

3.Place spam.csv in the project folder or update the path in the code.

4.Run the project:
python spam_detector.py

Example Usage:
my_text = "Congratulations! You've won a free ticket to the Bahamas. Click here to claim your prize."
prediction = model.predict(vectorizer.transform([my_text]))
print(prediction)  # 1 = spam, 0 = ham

Visualizations:

-Average message length by type (spam vs ham)
-Confusion matrix
-Most indicative words for spam and ham messages

Future Improvements:

-Add message length as an additional feature to improve model accuracy
-Try other machine learning models (Random Forest, Naive Bayes, etc.)
-Create an interactive interface using Streamlit or similar

## Author
Michał Proskurnicki – [GitHub](https://github.com/michalproskurnicki)  
2st-year student, Informatics and Econometrics, Poznań University of Economics
