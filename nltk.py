# Install necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
  if isinstance(text, str):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)
  else:
    return ""

# Load the train and test data with the specified encoding
train_data = pd.read_csv(r'C:\Users\91861\Downloads\train1.csv', encoding='latin-1')
test_data = pd.read_csv(r'C:\Users\91861\Downloads\test1.csv', encoding='latin-1')

# Fill missing values with empty strings
train_data['crimeaditionalinfo'].fillna("", inplace=True)
test_data['crimeaditionalinfo'].fillna("", inplace=True)

# Apply preprocessing to the text columns
train_data['cleaned_text'] = train_data['crimeaditionalinfo'].apply(preprocess_text)
test_data['cleaned_text'] = test_data['crimeaditionalinfo'].apply(preprocess_text)

print("Preprocessed Train Data Sample:")
print(train_data.head())
print("Preprocessed Test Data Sample:")
print(test_data.head())

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data['cleaned_text'])
X_test = vectorizer.transform(test_data['cleaned_text'])

# Use 'category' for classification
y_train = train_data['category']
y_test = test_data['category']

# Initialize and train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

# Unique values in 'category'
unique_categories = train_data['category'].unique()
print("Unique Categories in Train Data:")
print(unique_categories)

# Count of unique values for 'category'
category_counts = train_data['category'].value_counts()
print("\nCategory Counts in Train Data:")
print(category_counts)

# Unique values in 'sub_category'
unique_sub_categories = train_data['sub_category'].unique()
print("\nUnique Sub-Categories in Train Data:")
print(unique_sub_categories)

# Count of unique values for 'sub_category'
sub_category_counts = train_data['sub_category'].value_counts()
print("\nSub-Category Counts in Train Data:")
print(sub_category_counts)

