from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from nltk.tokenize import RegexpTokenizer
from scipy.spatial.distance import cosine
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

tfidf = None
song_rep = None
stop_words = None
df = None

def initialize_lsa_system():
    print("initializing")
    global tfidf, song_rep, stop_words, df
    df = pd.read_csv("clean_song_data.csv")
    df = df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis='columns')
    df['lyrics'].fillna('', inplace=True)
    tfidf = TfidfVectorizer()
    print("training")
    doc_term_matrix = tfidf.fit_transform(df['lyrics']).T
    u, s, vt = svds(doc_term_matrix, k=1000)
    song_rep = np.dot(np.diag(s), vt).T
    stop_words = nltk.corpus.stopwords.words()
    print("model complete")

def preprocess_query(query_lyrics):
    tokenizer = RegexpTokenizer(r'[a-z]+')
    words = tokenizer.tokenize(query_lyrics.lower())
    sb = []
    p_stem = PorterStemmer()
    for word in words:
        if word not in stop_words:
            sb.append(p_stem.stem(word))
    return ' '.join(sb)

def lsa_query_rep(query):
    query = preprocess_query(query)
    query_rep = [tfidf.vocabulary_.get(w, -1) for w in query.split()]
    query_rep = [q for q in query_rep if q != -1]
    if query_rep:
        query_rep = np.mean(song_rep[query_rep], axis=0)
        return query_rep
    return None

def find_similar_songs(query, num_songs=10):
    query_rep = lsa_query_rep(query)
    if query_rep is not None:
        query_song_cos_dist = [cosine(query_rep, s_rep) for s_rep in song_rep]
        query_song_sort_idx = np.argsort(np.array(query_song_cos_dist))
        similar_songs = []
        i = 0
        seen_titles = set()
        while num_songs > 0 and i < len(query_song_sort_idx):
            df_idx = query_song_sort_idx[i]
            title = df['title'][df_idx]
            if query_song_cos_dist[df_idx] != 0 and title not in seen_titles:
                similar_songs.append(["Title: ", df['title'][df_idx]," By: ", df['artist_name'][df_idx]])
                num_songs -= 1
                seen_titles.add(title)
            i += 1
        return similar_songs
    return []
 
@app.route('/submit-lyrics', methods=['POST'])
def submit_lyrics():
    data = request.json
    lyrics = data['lyrics']
    print("Received lyrics:", lyrics)
    recommendations = find_similar_songs(lyrics)
    print(recommendations)
    return jsonify({"message": "Lyrics received", "recommendations": recommendations}), 200

if __name__ == '__main__':
    initialize_lsa_system() 
    app.run(debug=True)
