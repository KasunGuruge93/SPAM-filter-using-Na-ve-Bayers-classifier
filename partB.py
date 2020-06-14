import string

from nltk import PorterStemmer as Stemmer
from nltk.corpus import stopwords

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB  #chose Naive bayes classifier for text classification
from sklearn.metrics import classification_report,confusion_matrix

messages = pd.read_csv('spam.csv', encoding='latin-1')

messages.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)

messages = messages.rename(columns={'v1': 'class','v2': 'text'})

messages['length'] = messages['text'].apply(len)

messages.hist(column='length',by='class',bins=200, figsize=(10,4),color = "red")



def process(text):
    # lowercase it
    text = text.lower()
    # remove punctuation
    text = ''.join([t for t in text if t not in string.punctuation])   #pre-processing step 
    # remove stopwords(removal of commonly used words)
    text = [t for t in text.split() if t not in stopwords.words('english')]
    # stemming (reducing related words to a common stem)
    st = Stemmer()
    text = [st.stem(t) for t in text]
    # return token list
    return text




messages['text'].apply(process).head()


msg_train, msg_test, class_train, class_test = train_test_split(messages['text'],messages['class'],test_size=0.2)

pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=process)), # converts strings to integer counts
    ('tfidf',TfidfTransformer()), # converts integer counts to weighted TF-IDF scores
    ('classifier',MultinomialNB()) # train on TF-IDF vectors with Naive Bayes classifier
])

pipeline.fit(msg_train,class_train)

predictions = pipeline.predict(msg_test)



import seaborn as sns
sns.heatmap(confusion_matrix(class_test,predictions),annot=True)


#Spam
def detect_spam(s):
    return pipeline.predict([s])[0]
detect_spam("Call Germany for only 1 pence per minute! Call from a fixed line via access number 0844 861 85 85. No prepayment. Direct access! www.telediscount.co.uk")

#ham
def detect_spam(s):
    return pipeline.predict([s])[0]
detect_spam("Babe !!!! I LOVE YOU !!!! *covers your face in kisses*")

