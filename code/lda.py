import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os

def load_data(data_dir):
    def load_file(file_name):
        with open(os.path.join(data_dir, file_name), 'r') as file:
            return [line.strip() for line in file]

    train_data = load_file('train.data')
    train_labels = load_file('train.label')
    vocabulary = load_file('vocabulary.txt')
    
    return train_data, train_labels, vocabulary

def preprocess_data(train_data, vocabulary):
    vectorizer = CountVectorizer(vocabulary=vocabulary)
    X_train = vectorizer.fit_transform(train_data)
    return X_train

def run_lda(X_train, n_topics):
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(X_train)
    return lda

def main():
    data_dir = 'data/20newsgroup'
    train_data, train_labels, vocabulary = load_data(data_dir)
    
    X_train = preprocess_data(train_data, vocabulary)
    
    n_topics = 20  # You can change this number
    lda_model = run_lda(X_train, n_topics)
    
    # Print the top words per topic
    terms = np.array(vocabulary)
    for idx, topic in enumerate(lda_model.components_):
        print(f"Topic #{idx + 1}:")
        print(" ".join(terms[i] for i in topic.argsort()[:-11:-1]))

if __name__ == "__main__":
    main()
