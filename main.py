#%% Importing the libraries

import pandas as pd
import re
import nltk
# Install if necessary - nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

#%% Importing the dataset

df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

#%% Preprocessing the reviews

corpus = []
for i in range(len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

#%% Creating the bag of words

cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = df.iloc[:, -1].values

#%% Splitting the dataset

x_train, x_test, y_train, y_true = train_test_split(x, y, test_size=0.2)

#%% Naive Bayes Classifier

classifier = GaussianNB()
classifier.fit(x_train, y_train)

#%% Prediction

y_pred = classifier.predict(x_test)

#%% Model Evaluation

cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
