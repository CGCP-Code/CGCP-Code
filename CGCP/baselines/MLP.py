###################### Train models with seeds and merge unconfident predictions ###################
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import random
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load train and test data for status calculation
train_data = pd.read_csv("train2.csv")
train_data['date'] = pd.to_datetime(train_data['date'])

test_data = pd.read_csv("test2.csv")
test_data['date'] = pd.to_datetime(test_data['date'])

# Combine train and test data for historical lookup
all_data = pd.concat([train_data[['user_id', 'date']], test_data[['user_id', 'date']]], ignore_index=True)
all_data = all_data.sort_values(['user_id', 'date']).reset_index(drop=True)
all_unconfident_predictions = []

for SEED in range(924, 934):
    print(f"\n{'='*60}")
    print(f"RANDOM SEED: {SEED}")
    print(f"{'='*60}")

    # Set random seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load training data
    df = pd.read_csv("train2.csv")

    # User demographics and contextual information covariates
    cat_features = ["country_id", "age_range", "gender", "role", "land_holding", "show_feedback_prompt",
                    "geography_level2_id", "preferred_language_id", "has_past_recent",
                    "has_past", "receive_com_via_whatsapp", "specialization"]
    df[cat_features] = df[cat_features].fillna("missing").astype(str)

    # Numeric features
    num_features = ["num_past_message", "num_past_recent", "num_has_followup", "num_past_yd",
                    "pct_has_followup","time_gap"]
    tabular_features = cat_features + num_features

    # Split train/val by 0.9/0.1 randomly
    X, y = df[tabular_features], df["churn"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.10, random_state=42
    )

    # Address missing value and convert cat features into binaries
    preprocessor = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_features),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
        ]), num_features),
    ], remainder="drop")

    X_train, X_val = preprocessor.fit_transform(X_train), preprocessor.transform(X_val)

    # Create pytorch tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train.values, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val.values, dtype=torch.long, device=device)

    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)

    # Define a mlp network
    class MLPClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )

        def forward(self, x):
            return self.net(x)

    mlp_model = MLPClassifier(X_train_t.shape[1]).to(device)

    # Setting hyperparameters
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)


    for epoch in range(1, 4):
        mlp_model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = mlp_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        mlp_model.eval()
        with torch.no_grad():
            val_logits = mlp_model(X_val_t)
            preds = val_logits.argmax(dim=1).cpu().numpy()
            current_f1 = f1_score(y_val, preds, average="macro")

        print(f"Epoch {epoch:2d} loss={avg_loss:.4f} val_macro_F1={current_f1:.4f}")

        # still step LR scheduler
        scheduler.step(current_f1)

    # Load test data and make predictions
    df_test = pd.read_csv("test2.csv")
    df_test[cat_features] = df_test[cat_features].fillna("missing").astype(str)
    df_test[num_features] = df_test[num_features].astype(float)

    X_test = df_test[tabular_features]
    y_test = df_test["churn"].astype(int)

    X_test_np = preprocessor.transform(X_test)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32, device=device)

    # Use the model after final epoch (no 'best' model saved)
    mlp_model.eval()

    with torch.no_grad():
        logits = mlp_model(X_test_t)
        test_probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = test_probs.argmax(axis=1)

    test_macro_f1 = f1_score(y_test, preds, average="macro")
    test_confidences = test_probs.max(axis=1)
    df_test["mlp_pred"] = preds
    df_test["mlp_confidence"] = test_confidences

    # Filter rows with confidence <= 0.6
    confidence_threshold = 0.6
    low_conf_mask = test_confidences <= confidence_threshold
    low_conf_rows = df_test[low_conf_mask].copy()
    selected_columns = [
        "user_id", "date", "num_message", "has_past", "num_past_message",
        "combined_message", "combined_response", "conversation",
        "churn", "mlp_pred", "mlp_confidence"
    ]
    low_conf_rows = low_conf_rows[selected_columns]
    low_conf_rows['seed'] = SEED
    test_df = low_conf_rows.copy()
    test_df['date'] = pd.to_datetime(test_df['date'])

    # Calculate status
    def calculate_status(row):
        user_id = row['user_id']
        test_date = row['date']
        previous_rows = all_data[(all_data['user_id'] == user_id) & (all_data['date'] < test_date)]

        if len(previous_rows) > 0:
            previous_date = previous_rows['date'].max()
            time_diff = (test_date - previous_date).days

            if time_diff > 3:
                return 'inactive'
            else:
                return 'active'
        else:
            return 'new'

    test_df['status'] = test_df.apply(calculate_status, axis=1)
    all_unconfident_predictions.append(test_df)
    test_df.to_csv(f"unconfident_predictions_seed_{SEED}.csv", index=False)
