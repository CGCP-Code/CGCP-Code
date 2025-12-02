import pandas as pd
import openai
from tqdm import tqdm
import time
import os

# Load data with existing topics and patterns
df = pd.read_csv("data_with_signals_0.csv")

# df = df.sample(n=50, random_state=42)

def classify_followup(row):
    """Classify if current message is a follow-up to any message in history"""

    current_message = row['message']
    history = row['history']

    prompt = (
        "Determine if the current message shows feedback or engagement with chatbot suggestions from history.\n\n"
        "Return '1' if the current message:\n"
        "- Reports results, outcomes, or feedback from following chatbot's advice in history\n"
        "- Updates on actions taken based on previous chatbot suggestions\n\n"
        "Ignore 'EMPTY' entries.\n"
        "Return ONLY '1' or '0'.\n\n"
        f"<HISTORY>\n{history}\n</HISTORY>\n\n"
        f"<CURRENT_MESSAGE>\n{current_message}\n</CURRENT_MESSAGE>\n\n"
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
                return 0  # Default to 0 if unclear response

        except Exception as e:
            print(f"Attempt {attempt + 1}/3 failed: {e}")
            time.sleep(2)

    return 0  # Default to 0 if all attempts fail


tqdm.pandas(desc="Processing follow-up classifications")
df['engagement'] = df.progress_apply(classify_followup, axis=1)

# Save results
df.to_csv("data_with_signals_2.csv", index=False)

print(f"Total follow-up messages: {df['engagement'].sum()}")
print(f"Total non-follow-up messages: {(df['engagement'] == 0).sum()}")
print(f"Percentage follow-ups: {(df['engagement'].sum() / len(df)) * 100:.2f}%")
