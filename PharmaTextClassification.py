import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.externals import joblib

# Initialize NLTK and download necessary resources
nltk.download('punkt')
nltk.download('stopwords')

# Load English stopwords and extend with custom words
stop = stopwords.words('english')
stop.extend(['am', 'pm'])

#Function to preprocess the text
def preprocess_text(text):
    text = text.lower()
    text_clean = re.sub(r'[*\d@_!#$%^&*()<>?/\|;,"}{~:-]', '', text)
    text_clean = re.sub(r'[x|a|\.]{2,}', '', text_clean)
    text_clean = text_clean.split()
    ps = PorterStemmer()
    words = [ps.stem(word) for word in text_clean if word not in stop]
    cleaned_text = ' '.join(words)
    return cleaned_text

#Function to load preprocessed data to a csv file
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.drop(["Unnamed: 4"], axis=1, inplace=True)
    corpus = [preprocess_text(summary) for summary in data['Summary']]
    data_clean = data.drop(["Summary"], axis=1)
    data_clean['Summary'] = corpus
    return data_clean

#Function to train the Naive Bayes model with train data
def train_naive_bayes_model(X, y):
    cv = CountVectorizer()
    X_new = cv.fit_transform(X).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    
    return model, accuracy, cm, classification_rep

#Function to save the model
def save_model(model, model_filename):
    joblib.dump(model, model_filename)

# Load and preprocess main category data
main_data_clean = load_and_preprocess_data('pharmatext.csv')

# Train and evaluate main category model
main_model, main_accuracy, main_cm, main_classification_rep = train_naive_bayes_model(
    main_data_clean['Summary'], main_data_clean['Categories']
)
print("Main Category Model Accuracy:", main_accuracy)
print("Main Category Confusion Matrix:\n", main_cm)
print("Main Category Classification Report:\n", main_classification_rep)

# Save main category model
save_model(main_model, 'main_category_model.pkl')

# Load and preprocess subcategory data
sub_data_clean = load_and_preprocess_data('pharmaclean.csv')

# Train and evaluate subcategory model
sub_model, sub_accuracy, sub_cm, sub_classification_rep = train_naive_bayes_model(
    sub_data_clean['Summary'], sub_data_clean['Subcategories']
)
print("Subcategory Model Accuracy:", sub_accuracy)
print("Subcategory Confusion Matrix:\n", sub_cm)
print("Subcategory Classification Report:\n", sub_classification_rep)

# Save subcategory model
save_model(sub_model, 'subcategory_model.pkl')

# Application Example
test_text = 'mri results are out'     #You can also input a string of your own
test_summary = preprocess_text(test_text)
test_summary = [test_summary]

# Transform and predict with main category model
test_main = cv.transform(test_summary).toarray()
main_category_prediction = main_model.predict(test_main)
print(main_category_prediction)
main_category_probabilities = main_model.predict_proba(test_main)
print(main_category_probabilities)

# Transform and predict with subcategory model
test_sub = cv.transform(test_summary).toarray()
subcategory_prediction = sub_model.predict(test_sub)
print(subcategory_prediction)
subcategory_probabilities = sub_model.predict_proba(test_sub)
print(subcategory_probabilities)






