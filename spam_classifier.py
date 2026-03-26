import pandas as pd

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Show first 5 rows
print(data.head())
# Keep only required columns
data = data[['v1', 'v2']]

# Rename columns
data.columns = ['label', 'message']

# Convert label to numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

print(data.head())
from sklearn.feature_extraction.text import CountVectorizer

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])

y = data['label']

print(X.shape)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Split data (train + test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = MultinomialNB()

# Train model
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
msg = ["You won a free lottery!!!"]
msg_vec = vectorizer.transform(msg)

result = model.predict(msg_vec)

if result[0] == 1:
    print("Spam ❌")
else:
    print("Not Spam ✅")