# Install necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from imblearn.over_sampling import SMOTE

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

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

# Check class distribution
class_distribution = y_train.value_counts()
print("Class Distribution in y_train:")
print(class_distribution)

# Filter out classes with fewer than 5 samples
min_samples = 5
filtered_indices = y_train.isin(class_distribution[class_distribution >= min_samples].index)
X_train_filtered = X_train[filtered_indices]
y_train_filtered = y_train[filtered_indices]

# Define a custom sampling strategy to handle classes with very few samples
sampling_strategy = {label: min(5000, count * 2) for label, count in class_distribution.items() if count >= min_samples and count < 5000}

# Handle class imbalance using SMOTE with adjusted k_neighbors
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=min(min_samples, 5) - 1)
X_train_smote, y_train_smote = smote.fit_resample(X_train_filtered, y_train_filtered)

print("Class Distribution after SMOTE:")
print(Counter(y_train_smote))

# Initialize and train the model with GridSearchCV for hyperparameter tuning
param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_smote, y_train_smote)

# Best model from GridSearchCV
model = grid_search.best_estimator_

# Make predictions on the test data
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f'Best Model Parameters: {grid_search.best_params_}')
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

