import pandas as pd
import openai
from tqdm import tqdm
import time
import os

# Load data with existing topics and patterns
df = pd.read_csv("data_with_topics_pred_changed.csv")

# df = df.sample(n=50, random_state=42)

def classify_conversation(row):
    """Classify if conversation is about protecting crops from rain"""
    conversation = row['message']
    topic = "Nitrogen fertilizer usage"

    prompt = (
        f"Analyze this conversation and determine if the topic is '{topic}'.\n\n"
        "Instructions:\n"
        "- Return ONLY '1' if the topic is in the conversation.\n"
        "- Return ONLY '0' if the topic is not in the conversation\n"
        "- A topic is in the conversation only when its key term and subject is discussed\n"
        "- Base your decision strictly on the conversation content\n"
        "- Do not provide any explanation, just the number\n\n"
        "Examples:\n\n"
        "Example Topic: 'Disease prevention in potatoes'\n\n"
        "POSITIVE Example (return 1):\n"
        "'How can I prevent blight in my potato crops?'\n"
        "→ Classification: 1 (The conversation directly discusses preventing diseases in potatoes)\n\n"
        "NEGATIVE Example (return 0):\n"
        "'What diseases affect potatoes?'\n"
        "→ Classification: 0 (The conversation mentions potato diseases but does NOT discuss prevention methods)\n\n"
        f"Now analyze this conversation for the topic '{topic}':\n\n"
        f"<CONVERSATION>\n{conversation}\n</CONVERSATION>\n\n"
        "Classification (1 or 0):"
    )

    for attempt in range(3):
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5
            )

            response = resp.choices[0].message.content.strip()

            # Ensure we get a valid response
            if response in ['1', '0']:
                return int(response)
            else:
                return 0  # Default to 0 if unclear response

        except Exception as e:
            print(f"Attempt {attempt + 1}/3 failed: {e}")
            time.sleep(2)

    return 0  # Default to 0 if all attempts fail


# Apply classification
print("Classifying conversations for crop rain protection criteria...")
tqdm.pandas(desc="Processing classifications")
df['stay_38'] = df.progress_apply(classify_conversation, axis=1)

# Save results
df.to_csv("data_with_stay_38.csv", index=False)

print(f"Total positive classifications: {df['stay_38'].sum()}")
print(f"Total negative classifications: {(df['stay_38'] == 0).sum()}")
print(f"Percentage positive: {(df['stay_38'].sum() / len(df)) * 100:.2f}%")