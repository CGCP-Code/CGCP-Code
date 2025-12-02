################ Joint Training text and tabular variables with Llama 3-8B + LoRA ###############
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tqdm
import random


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)


MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 2
LR = 2e-4
BASE_OUTPUT_DIR = "./llama3_8b_lora_joint_models"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

RANDOM_SEEDS = [924, 925, 926, 927, 928, 929, 930, 931, 932, 933]

cat_features = [
    "country_id", "age_range", "gender", "role", "land_holding",
    "show_feedback_prompt", "geography_level2_id", "preferred_language_id",
    "has_past_recent", "has_past", "receive_com_via_whatsapp", "specialization"
]

num_features = [
    "num_past_message", "num_past_recent", "num_has_followup",
    "num_past_yd", "pct_has_followup", "time_gap"
]


class JointDataset(Dataset):
    def __init__(self, df, tokenizer, cat_features, cat_maps, num_features, scaler, max_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.cat_features = cat_features
        self.cat_maps = cat_maps
        self.num_features = num_features
        self.scaler = scaler
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        enc = self.tokenizer(
            row["conversation"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = enc.input_ids.squeeze(0)
        attention_mask = enc.attention_mask.squeeze(0)

        label = torch.tensor(row["churn"], dtype=torch.long)

        cat_indices = [
            self.cat_maps[f].get(str(row[f]), self.cat_maps[f]["<UNK>"])
            for f in self.cat_features
        ]
        cats = torch.tensor(cat_indices, dtype=torch.long)

        num_vals = np.array([row[f] for f in self.num_features]).reshape(1, -1)
        num_scaled = self.scaler.transform(num_vals).flatten()
        nums = torch.tensor(num_scaled, dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "cats": cats,
            "nums": nums,
            "labels": label
        }


class JointClassifierLoRA(nn.Module):
    def __init__(self, text_model_name, cat_features, cat_maps, num_features,
                 lora_r=8, lora_alpha=16, lora_dropout=0.1, cat_embed_dim=32, num_labels=2):
        super().__init__()

        base_model = AutoModelForCausalLM.from_pretrained(
            text_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            device_map="auto",
            token=HF_TOKEN
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )

        self.text_model = get_peft_model(base_model, lora_config)
        self.text_model.print_trainable_parameters()

        hidden_size = self.text_model.config.hidden_size

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(len(cat_maps[f]), cat_embed_dim)
            for f in cat_features
        ])

        total_cat_dim = len(cat_features) * cat_embed_dim
        total_num_dim = len(num_features)

        self.tabular_mlp = nn.Sequential(
            nn.Linear(total_cat_dim + total_num_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(hidden_size + 64, num_labels)

        if torch.cuda.is_available():
            self.cat_embeddings.cuda()
            self.tabular_mlp.cuda()
            self.classifier.cuda()

    def forward(self, input_ids, attention_mask, cats, nums, labels=None, class_weights=None):

        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-9)
        text_feat = sum_embeddings / denom

        cat_feats = [emb(cats[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_feat = torch.cat(cat_feats, dim=1)

        tabular_input = torch.cat([cat_feat, nums], dim=1)
        tabular_processed = self.tabular_mlp(tabular_input)

        combined = torch.cat([text_feat, tabular_processed], dim=1)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            if class_weights is not None:
                loss = nn.functional.cross_entropy(logits, labels, weight=class_weights)
            else:
                loss = nn.functional.cross_entropy(logits, labels)

        return loss, logits


def train_with_seed(seed, df, test_df):
    print(f"\n{'=' * 80}")
    print(f"Training with Seed {seed}")
    print(f"{'=' * 80}\n")

    set_random_seed(seed)

    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"seed_{seed}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df, val_df = train_test_split(
        df, test_size=0.1, stratify=df["churn"], random_state=42
    )

    class_weights_array = compute_class_weight(
        "balanced",
        classes=np.unique(train_df["churn"]),
        y=train_df["churn"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor(class_weights_array, dtype=torch.float32, device=device)

    cat_maps = {}
    for feat in cat_features:
        uniques = sorted(train_df[feat].unique())
        mapping = {v: i for i, v in enumerate(uniques)}
        mapping["<UNK>"] = len(uniques)
        cat_maps[feat] = mapping

    scaler = StandardScaler()
    scaler.fit(train_df[num_features].values)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = JointDataset(train_df, tokenizer, cat_features, cat_maps, num_features, scaler, MAX_LENGTH)
    val_ds = JointDataset(val_df, tokenizer, cat_features, cat_maps, num_features, scaler, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = JointClassifierLoRA(
        MODEL_NAME, cat_features, cat_maps, num_features,
        lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT
    )

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    total_steps = len(train_loader) * NUM_EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cats = batch["cats"].to(device)
            nums = batch["nums"].to(device)
            labels = batch["labels"].to(device)

            loss, _ = model(
                input_ids, attention_mask, cats, nums,
                labels=labels, class_weights=class_weights
            )

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Skipping batch (NaN detected)")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch}: avg training loss = {epoch_loss / len(train_loader):.4f}")

        model.eval()
        preds_all, labels_all = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                cats = batch["cats"].to(device)
                nums = batch["nums"].to(device)
                labels = batch["labels"].to(device)

                _, logits = model(input_ids, attention_mask, cats, nums)
                preds = torch.argmax(logits, dim=1)
                preds_all.extend(preds.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

        val_f1 = f1_score(labels_all, preds_all, average="macro")
        print(f"Epoch {epoch}: validation macro-F1 = {val_f1:.4f}")
        print(classification_report(labels_all, preds_all, digits=4))

    model.eval()
    test_ds = JointDataset(test_df, tokenizer, cat_features, cat_maps, num_features, scaler, MAX_LENGTH)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cats = batch["cats"].to(device)
            nums = batch["nums"].to(device)
            labels = batch["labels"].to(device)

            _, logits = model(input_ids, attention_mask, cats, nums)
            preds = torch.argmax(logits, dim=1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    from sklearn.metrics import precision_recall_fscore_support
    macro_f1 = f1_score(labels_all, preds_all, average="macro")
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_all, preds_all, average=None, labels=[0, 1]
    )
    class1_precision = precision[1]
    class1_recall = recall[1]
    class1_f1 = f1[1]

    print(f"\nSeed {seed} — Test macro-F1: {macro_f1:.4f}")
    print(f"Class 1 — Precision: {class1_precision:.4f}, Recall: {class1_recall:.4f}, F1: {class1_f1:.4f}")
    print(classification_report(labels_all, preds_all, digits=4))

    return {
        "macro_f1": macro_f1,
        "class1_precision": class1_precision,
        "class1_recall": class1_recall,
        "class1_f1": class1_f1
    }


if __name__ == "__main__":
    df = pd.read_csv("train2.csv").fillna({"conversation": ""})
    df[cat_features] = df[cat_features].fillna("missing").astype(str)

    for f in num_features:
        df[f] = pd.to_numeric(df[f], errors='coerce')
        df[f] = df[f].fillna(df[f].median())

    df = df[["conversation", "churn"] + cat_features + num_features]

    test_df = pd.read_csv("test2.csv").fillna({"conversation": ""})
    test_df[cat_features] = test_df[cat_features].fillna("missing").astype(str)

    for f in num_features:
        test_df[f] = pd.to_numeric(test_df[f], errors='coerce')
        test_df[f] = test_df[f].fillna(df[f].median())

    test_df = test_df[["conversation", "churn"] + cat_features + num_features]

    results = {}
    for seed in RANDOM_SEEDS:
        results[seed] = train_with_seed(seed, df, test_df)

    macro_list = [results[s]['macro_f1'] for s in RANDOM_SEEDS]
    p_list = [results[s]['class1_precision'] for s in RANDOM_SEEDS]
    r_list = [results[s]['class1_recall'] for s in RANDOM_SEEDS]
    f1_list = [results[s]['class1_f1'] for s in RANDOM_SEEDS]

    print("\nPer-seed metrics:")
    for seed in RANDOM_SEEDS:
        print(f"{seed}: Macro-F1={results[seed]['macro_f1']:.4f}, "
              f"P={results[seed]['class1_precision']:.4f}, "
              f"R={results[seed]['class1_recall']:.4f}, "
              f"F1={results[seed]['class1_f1']:.4f}")

    print("\nAggregate statistics:")
    print(f"Macro-F1: mean={np.mean(macro_list):.4f}, std={np.std(macro_list):.4f}")
    print(f"Class1 Precision: mean={np.mean(p_list):.4f}")
    print(f"Class1 Recall:    mean={np.mean(r_list):.4f}")
    print(f"Class1 F1:        mean={np.mean(f1_list):.4f}")

    results_df = pd.DataFrame({
        "seed": RANDOM_SEEDS,
        "macro_f1": macro_list,
        "class1_precision": p_list,
        "class1_recall": r_list,
        "class1_f1": f1_list
    })
    results_df.to_csv(os.path.join(BASE_OUTPUT_DIR, "seed_results.csv"), index=False)
    print("\nSaved results to:", os.path.join(BASE_OUTPUT_DIR, "seed_results.csv"))