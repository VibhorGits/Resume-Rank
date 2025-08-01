import os
import re

import nltk
from PyPDF2 import PdfReader
from nltk import pos_tag  # method for calling averaged_perceptron_tagger

from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer

lem = WordNetLemmatizer()  # reduces words to dict base form

nltk.download('stopwords')
nltk.download('punkt')  # for splitting text
nltk.download('averaged_perceptron_tagger')  # ML model for attaching parts of speech to each word of sentence

stp = stopwords.words('english')
addw = ['job', 'description', 'qualification', 'responsibility', 'requirement', 'skills']
stp.extend(addw)


def extract(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text()
    return preprocess(text)


def preprocess(text):
    text = text.lower()
    tk = custom(text)  # tokenizing
    filtered = [w for w in tk if w not in stp and w != '.']  # remove stopwords
    stemd = [' '.join(lem.lemmatize(w) for w in filtered)]  # reduce to base form
    return stemd


def custom(text):
    pattern = r'\w*[a-zA-Z\d.+*-/#]+\w*'
    tokens = re.findall(pattern, text)
    # print(tokens)
    return tokens


def keyextract(text):
    tk = nltk.word_tokenize(text)  # tokenize text to words
    tag = nltk.pos_tag(tk)
    kw = ' '.join(word for word, pos in tag if pos.startswith('NN'))
    return kw


cv = CountVectorizer(tokenizer=custom, stop_words='english')  # text--> Numerical data


def func(jobd, pdff):
    jobd = keyextract(jobd)
    jobd = preprocess(jobd)
    # print(jobd)
    jobv = cv.fit_transform(jobd)
    # print(jobv.toarray())

    rank = []

    for pdf in pdff:
        restext = extract(pdf)
        # print(restext)
        fname = os.path.basename(pdf)
        resvec = cv.transform(restext)
        x = cosine_similarity(jobv, resvec)[0][0]
        rank.append((fname, x))

    rank.sort(key=lambda x: x[1], reverse=True)
    return rank
