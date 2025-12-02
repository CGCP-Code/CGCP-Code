###################### Topic clustering + test topic counts + F1 tweaks ####################

import pandas as pd
import re
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# ─── Device setup ─────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─── Helper: parse topics string ──────────────────────────────────────────────
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

print("\n[Train] Loading train_with_topics.csv ...")
df = pd.read_csv("train_with_topics.csv")
print(f"[Train] Rows: {len(df)}  | churn=0: {(df['churn'] == 0).sum()}  | churn=1: {(df['churn'] == 1).sum()}")

# Parse topics for all rows
print("[Train] Parsing topics column...")
df['topics_parsed'] = df['topics'].apply(parse_topics)

# Build set of unique topics
print("[Train] Collecting unique topics...")
all_topics = set()
for topics_list in tqdm(df['topics_parsed'], desc="[Train] Building unique topic set"):
    all_topics.update(topics_list)

all_topics = sorted(list(all_topics))
print(f"[Train] Unique topics: {len(all_topics)}")

# Load sentence-transformer model
print("[Train] Loading sentence-transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print(f"[Train] Model on device: {device}")

# Embeddings for all topics
print("[Train] Encoding topics...")
batch_size = 64 if device == 'cuda' else 32
topic_embeddings = model.encode(all_topics, show_progress_bar=True, batch_size=batch_size)

# Similarity / distance matrix
print("[Train] Computing similarity / distance matrices...")
similarity_matrix = cosine_similarity(topic_embeddings)
distance_matrix = 1 - similarity_matrix

# Agglomerative clustering
print("[Train] Running hierarchical clustering...")
clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.2,
    metric='precomputed',
    linkage='average'
)
cluster_labels = clustering.fit_predict(distance_matrix)
print(f"[Train] Number of clusters: {len(set(cluster_labels))}")

# Collect churn values for each raw topic
print("[Train] Collecting churn labels per topic...")
topic_all_churn_values = defaultdict(list)
for topics_list, churn in tqdm(
    zip(df['topics_parsed'], df['churn']),
    total=len(df),
    desc="[Train] Gathering churn values"
):
    for topic in topics_list:
        topic_all_churn_values[topic].append(churn)

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

# Build clusters and pick representatives
print("[Train] Building clusters and representatives...")
clusters_dict = defaultdict(list)
topic_to_index = {topic: idx for idx, topic in enumerate(all_topics)}

for topic, label in zip(all_topics, cluster_labels):
    clusters_dict[label].append(topic)

topic_mapping = {}
representative_to_subtopics = {}
representative_to_subtopics_with_summary = {}

for label, topics_in_cluster in clusters_dict.items():
    # Pick a "central" topic as representative
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

print(f"[Train] Clusters formed: {len(clusters_dict)}")
print("[Train] Showing up to 20 largest clusters (if size > 1):")
sorted_clusters = sorted(clusters_dict.items(), key=lambda x: len(x[1]), reverse=True)

for i, (label, members) in enumerate(sorted_clusters[:20]):
    if len(members) > 1:
        representative = None
        for topic in members:
            if topic_mapping[topic] == topic:
                representative = topic
                break

        print(f"\nCluster {i+1}  | size={len(members)}")
        print(f"  Representative: {representative!r}")
        print("  Example members with churn counts:")
        for member in members[:5]:
            stats = topic_churn_stats.get(member, {'count_0': 0, 'count_1': 0})
            print(f"    - {member} (0:{stats['count_0']}, 1:{stats['count_1']})")

# Map topics to representatives
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

print("\n[Train] Applying topic mapping to rows...")
df['topics_mapped'] = [
    map_topics_list(topics_list, topic_mapping)
    for topics_list in tqdm(df['topics_parsed'], desc="[Train] Mapping topics")
]

# Count cluster representatives by churn
print("[Train] Counting representative topics by churn...")
topic_count_churn_0 = Counter()
topic_count_churn_1 = Counter()

for topics_list, churn in tqdm(
    zip(df['topics_mapped'], df['churn']),
    total=len(df),
    desc="[Train] Counting appearances"
):
    unique_topics = set(topics_list)

    if churn == 0:
        for topic in unique_topics:
            topic_count_churn_0[topic] += 1
    elif churn == 1:
        for topic in unique_topics:
            topic_count_churn_1[topic] += 1

# Build analysis table
print("[Train] Building topic_analysis_all dataframe...")
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

print("[Train] Saving topic_analysis_all.csv and train_with_mapped_topics.csv ...")
topic_analysis_df.to_csv("topic_analysis_all.csv", index=False)
df.to_csv("train_with_mapped_topics.csv", index=False)
print("[Train] Done with train clustering / analysis.")


# PART 2: Count topic occurrences

print("\n[Test] Counting topic occurrences with similarity...")

# Read filtered topics (manually filtered from topic_analysis_all.csv)
filtered_topics_df = pd.read_csv("topic_analysis_filtered.csv")
print(f"[Test] Filtered topics: {len(filtered_topics_df)}")

# Read the test data
test_df = pd.read_csv("test_with_topics.csv")
print(f"[Test] Test rows: {len(test_df)}")

# Parse topics in test set
print("[Test] Parsing topics...")
test_df['topics_parsed'] = test_df['topics'].apply(parse_topics)

# Unique topics in test set
print("[Test] Collecting unique topics from test set...")
all_test_topics = set()
for topics_list in test_df['topics_parsed']:
    all_test_topics.update(topics_list)
all_test_topics = sorted(list(all_test_topics))
print(f"[Test] Unique test topics: {len(all_test_topics)}")

# Embeddings for filtered topics and test topics
print("[Test] Encoding filtered topics...")
filtered_topics_list = filtered_topics_df['topic'].tolist()
batch_size = 64 if device == 'cuda' else 32
filtered_embeddings = model.encode(filtered_topics_list, show_progress_bar=True, batch_size=batch_size)

print("[Test] Encoding test topics...")
test_embeddings = model.encode(all_test_topics, show_progress_bar=True, batch_size=batch_size)

# Similarity matrix
print("[Test] Computing similarity matrix...")
similarity_matrix = cosine_similarity(filtered_embeddings, test_embeddings)

# Threshold: same as train clustering (distance 0.2 -> similarity 0.8)
distance_threshold = 0.2
similarity_threshold = 1 - distance_threshold
print(f"[Test] Similarity threshold: {similarity_threshold:.3f}")

# Mapping from test topic to index
test_topic_to_idx = {topic: idx for idx, topic in enumerate(all_test_topics)}

# Count matches
results = []
test_df['match'] = 0

print("[Test] Counting matches for each filtered topic across test rows...")
for idx, row in tqdm(
    filtered_topics_df.iterrows(),
    total=len(filtered_topics_df),
    desc="[Test] Processing filtered topics"
):
    target_topic = row['topic']
    count_0 = 0
    count_1 = 0

    target_similarities = similarity_matrix[idx]

    for test_idx, test_row in test_df.iterrows():
        topics_list = test_row['topics_parsed']
        churn_value = test_row['churn']

        match_found = False
        for topic in topics_list:
            if topic in test_topic_to_idx:
                topic_idx = test_topic_to_idx[topic]
                similarity = target_similarities[topic_idx]

                if similarity >= similarity_threshold:
                    match_found = True
                    break

        if match_found:
            test_df.at[test_idx, 'match'] = 1
            if churn_value == 1:
                count_1 += 1
            elif churn_value == 0:
                count_0 += 1

    results.append({
        'topic': target_topic,
        'count_0': count_0,
        'count_1': count_1,
        'total': count_0 + count_1
    })

results_df = pd.DataFrame(results)

total_count_0 = results_df['count_0'].sum()
total_count_1 = results_df['count_1'].sum()
total_overall = results_df['total'].sum()

print("\n[Test] Aggregated counts over all filtered topics:")
print(f"  count_0: {total_count_0}")
print(f"  count_1: {total_count_1}")
print(f"  total:   {total_overall}")

print("[Test] Saving test_topic_counts.csv and updated test_with_topics.csv ...")
results_df.to_csv("test_topic_counts.csv", index=False)
test_df.to_csv("test_with_topics.csv", index=False)
print(f"[Test] Rows with match=1: {(test_df['match'] == 1).sum()}")
print(f"[Test] Rows with match=0: {(test_df['match'] == 0).sum()}")


# PART 3: F1 score calculation

print("\n[Eval] Calculating F1 with modified predictions...")

# Reload test (with match column) and MLP predictions
test_df = pd.read_csv("test_with_topics.csv")
mlp_df = pd.read_csv("mlp_test_predictions.csv")

y_true = test_df['churn']

# Original predictions
print("[Eval] Original MLP predictions:")
original_pred = mlp_df['mlp_pred'].copy()
f1_original = f1_score(y_true, original_pred, pos_label=1)
precision_original = precision_score(y_true, original_pred, pos_label=1)
print(f"  F1 (class 1): {f1_original:.4f}")
print(f"  Precision (class 1): {precision_original:.4f}")

# Step 1: set pred=1 where confidence <= 0.6
print("[Eval] Step 1: force pred=1 where confidence <= 0.6")
modified_pred_1 = mlp_df['mlp_pred'].copy()
modified_pred_1[mlp_df['confidence'] <= 0.6] = 1
f1_step1 = f1_score(y_true, modified_pred_1, pos_label=1)
precision_step1 = precision_score(y_true, modified_pred_1, pos_label=1)
print(f"  F1 (class 1): {f1_step1:.4f}")
print(f"  Precision (class 1): {precision_step1:.4f}")

# Step 2: starting from step 1, set pred=0 where confidence <= 0.6 AND match == 1
print("[Eval] Step 2: among low-confidence rows, flip to 0 if match==1")
modified_pred_2 = modified_pred_1.copy()
modified_pred_2[(mlp_df['confidence'] <= 0.6) & (test_df['match'] == 1)] = 0
f1_step2 = f1_score(y_true, modified_pred_2, pos_label=1)
precision_step2 = precision_score(y_true, modified_pred_2, pos_label=1)
recall_step2 = recall_score(y_true, modified_pred_2, pos_label=1)

print(f"  Precision (class 1): {precision_step2:.4f}")
print(f"  Recall    (class 1): {recall_step2:.4f}")
print(f"  F1        (class 1): {f1_step2:.4f}")

print("\n[Done] All steps finished.")
