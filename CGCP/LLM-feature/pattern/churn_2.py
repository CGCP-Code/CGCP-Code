import pandas as pd
import openai
from tqdm import tqdm
import time
import os

# Load data with conversation column
df = pd.read_csv("data_churn_check.csv")

# df = df.sample(n=50, random_state=42)

def classify_error(row):
    """Classify if responses in conversation properly address all questions"""
    conversation = row['conversation']

    prompt = (
        "Analyze the conversation containing Question-Response pairs.\n\n"
        "Evaluate if the user receives a long and complex answer from the AI agent that may be overwhelming, confusing, or too technical for them.\n\n"
        "Return '0' if:\n"
        "- No response is complex.\n\n"
        "Return '1' if:\n"
        "- Any response is complex.\n\n"
        "Return ONLY '1' or '0', no explanation.\n\n"
        f"<CONVERSATION>\n{conversation}\n</CONVERSATION>\n\n"
        "Classification:"
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
                return 1  # Default to 1 (error) if unclear response

        except Exception as e:
            print(f"Attempt {attempt + 1}/3 failed: {e}")
            time.sleep(2)

    return 1  # Default to 1 (error) if all attempts fail


tqdm.pandas(desc="Processing error classifications")
df['complex'] = df.progress_apply(classify_error, axis=1)

# Save results
df.to_csv("churn_2.csv", index=False)

print(f"Total conversations with errors: {df['complex'].sum()}")
print(f"Total conversations without errors: {(df['complex'] == 0).sum()}")
print(f"Percentage with errors: {(df['complex'].sum() / len(df)) * 100:.2f}%")