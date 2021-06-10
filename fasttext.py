import fasttext.util
import pandas as pd
fasttext.util.download_model('tr', if_exists='ignore')
ft = fasttext.load_model('cc.tr.300.bin')
df = pd.read_csv('data.csv')

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from snowballstemmer import TurkishStemmer
turkStem = TurkishStemmer()

df['SUMMARY'] = df['SUMMARY'].astype(str)
nltk.download('stopwords')

summ_df = (df['SUMMARY'].apply(lambda x: ' '.join([turkStem.stemWord(word.lower())
                                                       for word in nltk.word_tokenize(x)
                                                       if word not in stopwords.words(
        "turkish") and word.isalpha()]))).to_frame()

summ_df.to_csv("summ_after_prep.csv", sep='\t', encoding='utf-8')

import numpy as npf
from gensim.models.wrappers import FastText

model = FastText.load_fasttext_format('cc.tr.300.bin')


item_vectors = pd.DataFrame()
item_desc = df['SUMMARY']

for index, value in item_desc.items():
    sentence = value
    print(index, sentence)
    splits = sentence.split()
    my_list = np.zeros((0, 300))
    for split in splits:
        vec = model.wv[split]
        arr = np.array(vec)
        my_list = np.vstack((my_list, arr))
    my_list = my_list.mean(axis=0)
    temp_df = pd.DataFrame(data=my_list)
    item_vectors = pd.concat([item_vectors, temp_df.T])


from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


item_vectors.fillna(item_vectors.mean(),inplace=True)
print(item_vectors.isna().sum().sum())

X = item_vectors.mask(np.isinf(item_vectors))
X.fillna(X.mean(axis=1))
X = X.dropna()
print(X.isna().sum().sum())

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1, 12))
visualizer.fit(item_vectors)  # Fit the data to the visualizer
visualizer.show()  # Finalize and render the figure

kmeans = KMeans(n_clusters=7, random_state=0).fit(item_vectors)
y_kmeans = kmeans.predict(item_vectors)
centers = kmeans.cluster_centers_
sse = kmeans.inertia_
print(sse)
print(y_kmeans)

print(type(y_kmeans))

