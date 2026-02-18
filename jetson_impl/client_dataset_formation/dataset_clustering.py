import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import sys
from datasets import load_dataset
import json
from sklearn.decomposition import PCA
import random

dt_name = sys.argv[1]
dt_split = sys.argv[2]
classification_column = sys.argv[3]

batch_size = 64

def find_optimal_clusters(embeddings, max_k=25):
    print(f"Calculating inertia for k=2 to {max_k}...")
    inertias = []
    K = range(5, max_k + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)
        
    # Geometric Elbow Method
    x_points = np.array(list(K))
    y_points = np.array(inertias)
    p1 = np.array([x_points[0], y_points[0]])
    p2 = np.array([x_points[-1], y_points[-1]])
    
    distances = []
    for i in range(len(x_points)):
        p0 = np.array([x_points[i], y_points[i]])
        numerator = np.abs((p2[1] - p1[1]) * p0[0] - (p2[0] - p1[0]) * p0[1] + p2[0] * p1[1] - p2[1] * p1[0])
        denominator = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
        distances.append(numerator / denominator)
    
    optimal_k = x_points[np.argmax(distances)]
    print(f"Optimal number of clusters determined: {optimal_k}")
    return optimal_k


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- Running on {device.upper()} ---")

# Load
try:
    if dt_name == "allenai/winogrande":
        dt = load_dataset(dt_name, "winogrande_xl")
    elif dt_name == "ehovy/race":
        dt = load_dataset(dt_name, "all")
    elif dt_name == "aps/super_glue":
        dt = load_dataset(dt_name, "boolq")
    else:
        dt = load_dataset(dt_name, trust_remote_code=True)
except Exception as e:
    print(f"Error: {e}")
    exit()

dt=dt[dt_split]

if "+" in classification_column:
    columns=classification_column.split('+')
    classification_list=[]
    for con, q in zip(dt[columns[0]], dt[columns[1]]):
        classification_list.append(f"{con}\n{q}")
    
    sentences=[classification_list[i] for i in range(len(dt))]      
else:  
    sentences=[dt[classification_column][i] for i in range(len(dt))]

original_indices=list(range(len(dt)))

# Embed
print("Loading model all-mpnet-base-v2...")
model = SentenceTransformer('all-mpnet-base-v2', device=device)
embeddings = model.encode(sentences, batch_size=batch_size, show_progress_bar=True, device=device)

pca = PCA(n_components=0.95, random_state=42)
embeddings_reduced = pca.fit_transform(embeddings)
print(f"Dimensions reduced from {embeddings.shape[1]} to {embeddings_reduced.shape[1]}")

# Cluster
n_clusters = find_optimal_clusters(embeddings_reduced)
print(f"Clustering with k={n_clusters}...")
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
kmeans.fit(embeddings_reduced)
labels = kmeans.labels_

# Analysis
print("Analyzing terms...")
df_clustered = pd.DataFrame({'text': sentences, 'cluster': labels, 'original_index': original_indices})

# Get top terms via TF-IDF
grouped_text = df_clustered.groupby('cluster')['text'].apply(lambda x: ' '.join(x)).tolist()
domain_stop_words = list(TfidfVectorizer(stop_words='english').get_stop_words())
domain_stop_words.extend([
'make', 'use', 'using', 'used', 'clean', 'remove', 'prevent',
    'place', 'need', 'needed', 'prepare', 'best', 'way', 'good',
    'want', 'wanted', 'going', 'getting', 'cause', 'does', 'did', 
    'didn', 'wasn', 'doing', 'happen', 'looked', 'went', 'able',
    'instead', 'asked', 'decided', 'tried', 'thought', 'chose', 
    'got', 'took', 'new', 'old', 'better', 'easier', 'liked', 
    'ordered', 'fit', 'began', 'told', 'said', 'gave', 'feel', 
    'result',
    
    # Generic Entities
    'person', 'people', 'man', 'woman', 'child', 'likely', 'area', 
    'thing', 'things', 
    
    # Common Names (Social IQA + Winogrande + Generic)
    'john', 'james', 'billy', 'sarah', 'sally', 'kai', 'sydney', 
    'casey', 'robin', 'sasha', 'addison', 'kendall', 'remy', 
    'skylar', 'aubrey', 'jan', 'bailey', 'alex', 'jordan', 'ash', 
    'cameron', 'jesse', 'quinn', 'lee', 'riley', 'carson', 'taylor', 
    'austin', 'sam', 'tracy',
    
    # Winogrande Specific Names (Female)
    "elena", "patricia", "christine", "kayla", "lindsey", "tanya", 
    "victoria", "rachel", "megan", "mary", "katrina", "monica", 
    "julie", "amy", "lisa", "stephanie", "erica",

    # Winogrande Specific Names (Male)
    "joel", "kyle", "steven", "craig", "ian", "kevin", "ryan", 
    "neil", "logan", "benjamin", "samuel", "jason", "donald", 
    "derrick", "justin", "eric", "adam", "jeff", "brian"
])

print(f"DEBUG: Stop word list has {len(domain_stop_words)} words.", flush=True)
tfidf = TfidfVectorizer(stop_words=domain_stop_words, max_features=1000)
tfidf_matrix = tfidf.fit_transform(grouped_text)
feature_names = np.array(tfidf.get_feature_names_out())

# Write Output
dt_name=dt_name.replace('/', '_')
output_file = f"clusters_{dt_name}.json"
print(f"Writing results to {output_file}...")

cluster_indices_list=[]
top_words=[]

for i in range(n_clusters):
    cluster_indices = df_clustered[df_clustered['cluster'] == i]['original_index'].tolist()
    cluster_indices_list.append(cluster_indices)

    if i < len(grouped_text):
        top_indices = tfidf_matrix[i].toarray()[0].argsort()[::-1][:10]
        top_terms = feature_names[top_indices]
        top_words.append(top_terms.tolist())

data={
    "cluster_idx" : cluster_indices_list,
    "top_words" : top_words
}

with open("/files/impl_v2/client_dataset_formation/" + output_file, "w") as f:
    json.dump(data, f, indent=4) 

print(f"Done {dt_name}.")
