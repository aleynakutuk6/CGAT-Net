import argparse
import gensim.downloader
import json
import numpy as np
from tqdm import tqdm

# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('-td', '--train-dir', default=None, type=str, 
    help="Directory of the training experiment of single object classifier.")
args = parser.parse_args()


forbidden_words = {"the", "a", "I", "of", "an"}
change_maps = {"yoga:", "person"}

def get_wordlist_from_vocab(init_word, vocab):
    wordlist = []
    if init_word in change_maps:
        word = change_maps[init_word]
    else:
        word = init_word
        
    if word.lower() in vocab:
        wordlist = [word.lower()]
    elif word.lower().replace(" ", "-") in vocab:
        wordlist = [word.lower().replace(" ", "-")]
    elif word.lower().replace(" ", "-").replace("-", "") in vocab:
        wordlist = [word.lower().replace(" ", "-").replace("-", "")]
    else:
        for w in word.lower().replace("-", " ").split(" "):
            if w in vocab and w not in forbidden_words and len(w) > 2:
                wordlist.append(w)
                
    return wordlist

with open(f"{args.train_dir}/labels_info.json", "r") as f:
    data = json.load(f)
    
labels = [data["idx_to_label"][v] for v in data["idx_to_label"]]
    
print("Loading the model...")
# word2vec-google-news-300
# glove-wiki-gigaword-300
# glove-twitter-200
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')

nonexist_words = set()
sim_dict = {}

score_arr = np.full((len(labels), len(labels)), -1)

for i in range(len(labels) - 1):
    print()
    print(f"---> Doing: {i+1} / {len(labels)}")
    print()
    
    score_arr[i, i] = 1
    
    lbl1 = get_wordlist_from_vocab(labels[i], glove_vectors.key_to_index)
    if len(lbl1) == 0: 
        nonexist_words.add(labels[i])
        continue

    for j in range(i+1, len(labels)):
        lbl2 = get_wordlist_from_vocab(labels[j], glove_vectors.key_to_index)
        if len(lbl2) == 0: 
            nonexist_words.add(labels[j])
            continue
        
        scores, cnts = 0, 0
        for l1 in lbl1:
            for l2 in lbl2:
                scores += float(glove_vectors.similarity(l1, l2))
                cnts += 1
        score = scores / max(cnts, 1)
        
        print(f"[INFO] Similarity between {lbl1} and {lbl2}: {score}")
        ix = data["label_to_idx"][labels[i]]
        jx = data["label_to_idx"][labels[j]]
        score_arr[ix, jx] = score
        score_arr[jx, ix] = score
        
        sim_dict[f"{labels[i]} ## {labels[j]}"] = score
        
        
score_arr[-1, -1] = 1 
with open(f'{args.train_dir}/word_sim_score_mtx.npy', 'wb') as f:
    np.save(f, score_arr)
        
        
print("Non-existing words:", nonexist_words) 
 
with open(f"{args.train_dir}/sim_outs.json", "w") as f:
    json.dump(sim_dict, f)