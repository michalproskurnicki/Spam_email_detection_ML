import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load dataset
spam = pd.read_csv("spam.csv", encoding='latin-1')

# Drop unnecessary columns
spam = spam.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])

# Encode labels: ham = 0, spam = 1
spam["v1"] = spam["v1"].replace({"ham": 0, "spam": 1})
spam["v1"] = spam["v1"].astype(int)

# Clean text
spam["v2"] = spam["v2"].str.replace("[^\w\s]", "", regex=True).str.lower()
spam["v2"] = spam["v2"].str.replace("\s+", " ", regex=True)
spam["v2"] = spam["v2"].str.strip()

# Quick overview
print(spam.head())
print(spam.info())
print(spam.describe())

# Average length of messages by type
labels = ["Ham", "Spam"]
ham_messages = spam[spam["v1"] == 0]
spam_messages = spam[spam["v1"] == 1]

ham_text = ham_messages["v2"]
spam_text = spam_messages["v2"]

avg_length_ham = np.mean(ham_text.str.len())
avg_length_spam = np.mean(spam_text.str.len())

plt.bar(labels, [avg_length_ham, avg_length_spam], color=["blue", "red"])
plt.xlabel("Message Type")
plt.ylabel("Average Number of Characters")
plt.title("Average Message Length by Type")
plt.show()

# Prepare data for model
X = spam["v2"]
y = spam["v1"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Test model on new message
new_text = "Congratulations! You've won a free ticket to the Bahamas. Click here to claim your prize."
new_vec = vectorizer.transform([new_text])
prediction = model.predict(new_vec)
print("Prediction for new message:", prediction)  # 1 = spam, 0 = ham

# Top words for spam and ham
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]
words_weight = pd.DataFrame({'word': feature_names, 'weight': coefficients})

top_spam = words_weight.sort_values(by='weight', ascending=False).head(20)
top_ham = words_weight.sort_values(by='weight', ascending=True).head(20)

print("Top spam words:\n", top_spam)
print("Top ham words:\n", top_ham)

# Plot top spam words
plt.figure(figsize=(10,5))
plt.barh(top_spam["word"], top_spam["weight"])
plt.title("Top Spam Words")
plt.xlabel("Weight in Model")
plt.show()

# Plot top ham words
plt.figure(figsize=(10,5))
plt.barh(top_ham["word"], top_ham["weight"])
plt.title("Top Ham Words")
plt.xlabel("Weight in Model")
plt.show()
