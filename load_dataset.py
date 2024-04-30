"""
    Script that downloads the needed datasets taken from the MillionSongDataset as
    well as merging the data  with musiXmatch and Last.fm datasets to find corresponding
    lyric data. We used the following code to help guide our approach for correctly loading
    our data. https://github.com/DanieleMorotti/Songs-recommendation-from-lyrics/blob/main/create_dataset.py
"""
import os
import urllib.request
import shutil
import pandas as pd
from tqdm import tqdm
import sqlite3
import json

tqdm.pandas(desc="Constructing dataframe")

def download_file(src, dest, length):
    chunk_size = 1024

    with tqdm(total=length, unit="B", unit_scale=True, desc="Downloading file") as pbar:
        while True:
            chunk = src.read(chunk_size)

            if not chunk:
                break
            dest.write(chunk)
            pbar.update(len(chunk))

#Saves the data of type data_type into a file in dest_path of type mode
def save_file(data, dest_path, mode='csv', data_type='dataframe'):
    if data_type == "dataframe":
        if mode == "json":
            data.to_json(dest_path)
        elif mode == "csv":
            data.to_csv(dest_path, index=False)
    else:
        if mode == 'json':
            with open(dest_path, 'w+') as buff:
                buff.write(json.dumps(data))
        elif mode == 'txt':
            if type(data) != str:
                data = json.dumps(data)
            with open(dest_path, 'w+') as buff:
                buff.write(data)


#Converts mapping of idx:count to a list of words where each word is repeated count times
def tuple_to_bow(val_list, words_list):
    lyrics = ""

    for val in val_list:
        idx = val.split(':')
        lyrics += f"{words_list[int(idx[0])-1]} " * int(idx[1])

    return lyrics.rstrip()

#Merges the data with metadata to have the title, artist
#and lyrics into one dataframe.
def create_full_df(src, mxm_df, freq_words):
    metadata = sqlite3.connect(src)
    features = ['track_id', 'title', 'artist_name']

    df_meta = pd.read_sql_query("SELECT track_id, title, artist_name FROM songs", metadata)
    full_df = pd.merge(mxm_df, df_meta, on="track_id")
    metadata.close()

    tqdm.pandas(desc="converting word counts to lyrics")
    full_df['lyrics'] = full_df['words_count'].progress_apply(tuple_to_bow, words_list=freq_words)
    print(full_df)
    full_df = full_df.drop_duplicates('track_id', ignore_index=True)

    return full_df

#Iterates over all files needed for datasets and attempts to download it into the dest path
#Datasets parameter is dict where key is download url, and value is dest
def download_datasets(datasets):
    if not os.path.exists('downloads/'):
        os.mkdir('downloads')

    if type(datasets) != dict:
        print("datasets are not of type dict")
        return
    
    files = datasets.keys()

    for file in files:
        dest = os.path.join("downloads", datasets[file])

        if os.path.exists(dest):
            print("file {dest} already exists")
            continue
        with urllib.request.urlopen(file) as src, open(dest, "wb") as des:
            download_file(src, des, int(src.info()["Content-Length"]))

        if datasets[file].endswith('.zip'):
            shutil.unpack_archive(dest, "downloads")
        
    return "All files saved."

#Opens a file containing the song dataset and return the
#most frequent words and a list of words for each song
def prepare_song_dataset(path):
    with open(path, 'r') as buff:
        data = buff.readlines()[17:]
        most_freq_words = data.pop(0)[1:].split(',')
        most_freq_words = json.dumps(most_freq_words)

        cols = ['track_id', 'mxm_id', 'words_count']
        ids = []
        mxm_ids = []
        word_counts = []

        for line in tqdm(data, desc="Iterating rows of dataset"):
            values = line.split(',')
            ids.append(values[0])
            mxm_ids.append(values[1])
            word_counts.append(values[2:])

        final_df = pd.DataFrame({cols[0]: ids, cols[1]: mxm_ids, cols[2]: word_counts})
    return most_freq_words, final_df

#Cleans the data within the similar songs db and returns
#a list of tuples with song ID and similarity score
def aggregate_similar(str_list):
    split_list = str_list.split(',')
    res = []
    for i in range(0, len(split_list), 2):
        res.append((split_list[i], float(split_list[i+1])))
    return res

#Removes all the similar songs that are not in our main lyric dataset.
def check_evaluation_dataset(target, track_id_list):
    song_list = target.split(',')
    filtered = sorted(set(song_list).intersection(track_id_list), key=song_list.index)
    return ','.join(filtered)


#Prepares the evaluation dataset that contains a list of similar songs
#for each track ID
def prepare_evaluation_dataset(songs_data):
    conn = sqlite3.connect("downloads/lastfm_similar_songs.db")
    similar_songs = pd.read_sql_query("SELECT tid as track_id, target FROM similars_src", conn)
    conn.close()

    full_eval_df = pd.merge(songs_data, similar_songs, on="track_id")
    
    tqdm.pandas(desc="Creating the list of similar songs")
    reduced_df = full_eval_df.copy()
    reduced_df['target'] = reduced_df['target'].progress_apply(aggregate_similar)

    #Remove entries with less than 250 similar songs
    reduced_df = reduced_df[(reduced_df['target'].apply(len) >= 250)].reset_index(drop=True)
    clean_list = lambda lis: ",".join([val[0] for val in lis])
    reduced_df['target'] = reduced_df['target'].apply(clean_list)

    #Keep only songs that are in the lyrics dataset
    to_keep = songs_data.track_id
    tqdm.pandas(desc="Cleaning similar songs lists")
    reduced_df['target'] = reduced_df['target'].progress_apply(check_evaluation_dataset, track_id_list=to_keep)

    #Take at least 125 similar songs for each element
    reduced_df = reduced_df[reduced_df.target.str.split(',').apply(len) >= 125].reset_index(drop=True)

    to_keep2 = set(songs_data.track_id) - set(reduced_df.track_id)
    reduced_df['target'] = reduced_df['target'].progress_apply(check_evaluation_dataset, track_id_list=to_keep2)

    save_file(reduced_df, 'eval_similar_songs.csv', mode='csv', data_type='dataframe')

    return reduced_df, full_eval_df

#helps split the relevance score and song id into two columns
def split_sim_scores_id(row, mode=1):
    if mode==1:
        return ",".join(row.split(',')[::2])
    else:
        return ",".join(row.split(',')[1::2])
    
#Removes similar ids and scores across all songs that are not in main dataset
#Helps to clean the data
def check_gnn_dataset(row, track_id_list):
    target_list = row['similars'].split(',')
    scores_list = row['sim_scores'].split(',')
    ind_func = target_list.index

    curr = sorted(set(target_list).intersection(track_id_list), key=ind_func)
    scores_to_keep = [ind_func(track) for track in curr]
    pres_scores = list(map(scores_list.__getitem__, scores_to_keep))

    row['similars'] = ','.join(curr)
    row['sim_scores'] = ','.join(pres_scores)
    return row


#Creates the training and validation datasets used within the GNN
def create_gnn_dataset(similar_df, eval_df, val_size=0.2):
    #remove songs from eval_df  in similar_df
    gnn_df = similar_df[~similar_df['track_id'].isin(eval_df['track_id'])].reset_index(drop=True)

    gnn_df['similars'] = gnn_df['target'].progress_apply(split_sim_scores_id, mode=1)
    gnn_df['sim_scores'] = gnn_df['target'].progress_apply(split_sim_scores_id, mode=2)
    gnn_df = gnn_df.drop("target", axis=1)

    validation = gnn_df.sample(frac=val_size)
    training = gnn_df.drop(validation.index)

    #Remove songs not in the dataset from training and validation sets
    tqdm.pandas(desc="Cleaning the GNN dataset")
    training = training.progress_apply(check_gnn_dataset, axis=1, track_id_list=training['track_id'].tolist())
    validation = validation.progress_apply(check_gnn_dataset, axis=1, track_id_list=training['track_id'].tolist())

    #Removes songs with less than 0 similar songs
    training = training[training['similars'].str.split(",").apply(len) > 0]
    validation = validation[validation['similars'].str.split(",").apply(len) > 0]

    save_file(training, 'training_songs_gnn.csv', mode='csv', data_type='dataframe')
    save_file(validation, 'validation_songs_gnn.csv', mode='csv', data_type='dataframe')


def main():
    train_url = "http://millionsongdataset.com/sites/default/files/AdditionalFiles/mxm_dataset_train.txt.zip"
    song_metadata_url = "http://millionsongdataset.com/sites/default/files/AdditionalFiles/track_metadata.db"
    similar_song_url = "http://millionsongdataset.com/sites/default/files/lastfm/lastfm_similars.db"
    song_tags_url = "http://millionsongdataset.com/sites/default/files/lastfm/lastfm_tags.db"

    datasets = {train_url: "mxm_dataset.zip", song_metadata_url: "mxm_metadata.db", 
                similar_song_url: "lastfm_similar_songs.db", song_tags_url: "lastfm_tags.db"}
    
    download_datasets(datasets)
    print("Finished downloading datasets\n")
    
    print("Retrieving the songs dataset and words \n")
    most_freq_words, mxm_df = prepare_song_dataset("downloads/mxm_dataset_train.txt")
    save_file(most_freq_words, 'top_words.txt', mode="txt", data_type="list")
    most_freq_words = json.loads(most_freq_words)

    print("Merge mxm data with larger dataset to create the lyrics \n")
    full_df = create_full_df('downloads/mxm_metadata.db', mxm_df, most_freq_words)
    save_file(full_df, 'songs_data.csv', mode="csv", data_type="dataframe")

    print("Building the evaluation dataset \n")
    eval_df, larger_df = prepare_evaluation_dataset(full_df)

    #Remove the songs to test from training dataset and save the training dataset
    full_df = full_df[~full_df['track_id'].isin(eval_df['track_id'])].reset_index(drop=True)
    save_file(full_df, 'songs_data.csv', mode="csv", data_type="dataframe")

    print("Building the dataset for the GNN models \n")
    create_gnn_dataset(larger_df, eval_df)

    print("process finished \n")



if __name__ == "__main__":
    main()