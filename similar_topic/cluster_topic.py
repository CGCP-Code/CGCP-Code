####### Cluster topics from train_with_topics.csv and analyze churn patterns ##########
import pandas as pd
from tqdm import tqdm
import re
from collections import defaultdict, Counter
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\n" + "="*100)
print("LOADING DATA")
print("="*100)

# Read the dataset
df = pd.read_csv("train_with_topics.csv")
print(f"Loaded {len(df)} rows from train_with_topics.csv")
print(f"Rows with churn=0: {(df['churn']==0).sum()}")
print(f"Rows with churn=1: {(df['churn']==1).sum()}")

# Parse topics function
def parse_topics(topics_str):
    """Turn '1: topic1 2: topic2 3: topic3' into a list of topic strings."""
    if pd.isna(topics_str):
        return []
    topics = []
    matches = re.findall(r'\d+:\s*([^\d]+?)(?=\d+:|$)', str(topics_str))
    for match in matches:
        topic = match.strip()
        if topic:
            topics.append(topic)
    return topics

print("Parsing topics column...")
df['topics_parsed'] = df['topics'].apply(parse_topics)

print("Collecting unique topics...")
all_topics = set()
for topics_list in tqdm(df['topics_parsed'], desc="Building unique topic set"):
    all_topics.update(topics_list)

all_topics = sorted(list(all_topics))
print(f"Total unique topics: {len(all_topics)}")

print("Loading sentence-transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print(f"Model is on {device}")

print("Encoding all topics...")
batch_size = 64 if device == 'cuda' else 32
topic_embeddings = model.encode(all_topics, show_progress_bar=True, batch_size=batch_size)

print("Computing similarity matrix...")
similarity_matrix = cosine_similarity(topic_embeddings)

# similarity -> distance
distance_matrix = 1 - similarity_matrix

print("Running hierarchical clustering...")
clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.2,
    metric='precomputed',
    linkage='average'
)
cluster_labels = clustering.fit_predict(distance_matrix)

print(f"Got {len(set(cluster_labels))} clusters from {len(all_topics)} topics")

topic_all_churn_values = defaultdict(list)

print("Collecting churn labels for each topic...")
for topics_list, churn in tqdm(
    zip(df['topics_parsed'], df['churn']),
    total=len(df),
    desc="Collecting churn values"
):
    for topic in topics_list:
        topic_all_churn_values[topic].append(churn)

# Summary stats for each topic
topic_churn_stats = {}
for topic, churn_list in topic_all_churn_values.items():
    count_0 = churn_list.count(0)
    count_1 = churn_list.count(1)
    total = len(churn_list)
    topic_churn_stats[topic] = {
        'count_0': count_0,
        'count_1': count_1,
        'total': total,
        'all_values': churn_list
    }

clusters_dict = defaultdict(list)
topic_to_index = {topic: idx for idx, topic in enumerate(all_topics)}

for topic, label in zip(all_topics, cluster_labels):
    clusters_dict[label].append(topic)

topic_mapping = {}
representative_to_subtopics = {}
representative_to_subtopics_with_summary = {}

for label, topics_in_cluster in clusters_dict.items():

    if len(topics_in_cluster) == 1:
        representative = topics_in_cluster[0]
    else:
        cluster_indices = [topic_to_index[t] for t in topics_in_cluster]

        avg_similarities = []
        for topic in topics_in_cluster:
            topic_idx = topic_to_index[topic]
            similarities = [
                similarity_matrix[topic_idx][idx]
                for idx in cluster_indices
                if idx != topic_idx
            ]
            avg_sim = np.mean(similarities) if similarities else 0
            avg_similarities.append(avg_sim)

        representative = topics_in_cluster[np.argmax(avg_similarities)]

    representative_to_subtopics[representative] = sorted(topics_in_cluster)

    subtopics_with_summary = []
    for topic in sorted(topics_in_cluster):
        stats = topic_churn_stats.get(topic, {'count_0': 0, 'count_1': 0})
        summary_str = f"{topic} ({stats['count_0']}/{stats['count_1']})"
        subtopics_with_summary.append(summary_str)
    representative_to_subtopics_with_summary[representative] = subtopics_with_summary

    for topic in topics_in_cluster:
        topic_mapping[topic] = representative

print(f"Clustered into {len(clusters_dict)} groups")
print("Top 20 biggest clusters:")
sorted_clusters = sorted(clusters_dict.items(), key=lambda x: len(x[1]), reverse=True)

for i, (label, members) in enumerate(sorted_clusters[:20]):
    if len(members) > 1:
        representative = None
        for topic in members:
            if topic_mapping[topic] == topic:
                representative = topic
                break

        print(f"\n{i+1}. Cluster size: {len(members)}")
        print(f"   Representative: '{representative}'")
        print(f"   Sample members with churn counts:")
        for member in members[:5]:
            stats = topic_churn_stats.get(member, {'count_0': 0, 'count_1': 0})
            print(f"      - {member} (0:{stats['count_0']}, 1:{stats['count_1']})")

def map_topics_list(topics_list, mapping):
    """Map raw topic list to cluster representatives (deduplicated)."""
    mapped = [mapping.get(topic, topic) for topic in topics_list]
    seen = set()
    unique_mapped = []
    for topic in mapped:
        if topic not in seen:
            seen.add(topic)
            unique_mapped.append(topic)
    return unique_mapped

print("Applying topic mapping to data...")
df['topics_mapped'] = [
    map_topics_list(topics_list, topic_mapping)
    for topics_list in tqdm(df['topics_parsed'], desc="Mapping topics")
]

topic_count_churn_0 = Counter()
topic_count_churn_1 = Counter()

print("Counting topic appearances by churn (unique per row)...")
for topics_list, churn in tqdm(
    zip(df['topics_mapped'], df['churn']),
    total=len(df),
    desc="Counting appearances"
):
    unique_topics = set(topics_list)

    if churn == 0:
        for topic in unique_topics:
            topic_count_churn_0[topic] += 1
    elif churn == 1:
        for topic in unique_topics:
            topic_count_churn_1[topic] += 1

all_mapped_topics = set(topic_count_churn_0.keys()) | set(topic_count_churn_1.keys())

topic_analysis = []
for topic in all_mapped_topics:
    count_0 = topic_count_churn_0[topic]
    count_1 = topic_count_churn_1[topic]
    total = count_0 + count_1

    pct_0 = (count_0 / total * 100) if total > 0 else 0
    pct_1 = (count_1 / total * 100) if total > 0 else 0
    pct_difference = pct_0 - pct_1

    subtopics = representative_to_subtopics.get(topic, [topic])
    subtopics_str = " | ".join(subtopics)

    subtopics_with_summary = representative_to_subtopics_with_summary.get(
        topic, [f"{topic} (0/0)"]
    )
    subtopics_with_summary_str = " | ".join(subtopics_with_summary)

    cluster_size = len(subtopics)

    topic_analysis.append({
        'topic': topic,
        'cluster_size': cluster_size,
        'all_subtopics': subtopics_str,
        'subtopics_with_churn_summary': subtopics_with_summary_str,
        'count_churn_0': count_0,
        'count_churn_1': count_1,
        'total_appearances': total,
        'pct_churn_0': pct_0,
        'pct_churn_1': pct_1,
        'pct_difference': pct_difference,
    })

topic_analysis_df = pd.DataFrame(topic_analysis)

topic_analysis_df.to_csv("topic_analysis_all.csv", index=False)
print("Wrote topic_analysis_all.csv")

df.to_csv("train_with_mapped_topics.csv", index=False)
print("Wrote train_with_mapped_topics.csv")
