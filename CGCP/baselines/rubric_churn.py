import pandas as pd
import openai
from tqdm import tqdm
import time
import json
import re

df = pd.read_csv("test2.csv")
# df = df.sample(n=100)

def evaluate_conversation(conversation, max_retries=3):
    prompt = (
        "You are an AI evaluator. Given the conversation between the user and chatbot in the format "
        "{Question 1: ..., Response 1: ...}, {Question 2: ..., Response 2: ...}, etc., "
        "evaluate the following six aspects:\n\n"
        "1. Does the user repeat their query or request multiple times?\n"
        "2. Does the user point out an error, inconsistency, or inaccuracy in the chatbot's response?\n"
        "3. Does the user use a negative tone or words to express frustration, disappointment, anger, or disrespect towards the chatbot?\n"
        "4. Does the user change their topic or query abruptly?\n"
        "5. Does the user receive a generic, vague, irrelevant answer from the chatbot that does not address their specific needs, goals, or preferences?\n"
        "6. Does the user receive a long and complex answer from the chatbot that may be overwhelming, confusing, or too technical for them?\n\n"
        "IMPORTANT: Ignore any entries marked as 'EMPTY'. Only evaluate actual questions and responses.\n\n"
        f"Conversation:\n{conversation}\n\n"
        "Respond ONLY with a JSON object in this exact format:\n"
        '{"repetition": "Yes", "errors": "No", "negative": "Yes", "switch": "No", "irrelevant": "Yes", "complex": "No"}\n\n'
        "Do not include any other text, explanation, or markdown formatting."
    )

    for attempt in range(max_retries):
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            answer = resp.choices[0].message.content.strip()

            # Try to extract JSON from the response (handle markdown code blocks)
            json_match = re.search(r'\{[^}]+\}', answer)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = answer

            # Parse JSON response
            try:
                result = json.loads(json_str)
                return {
                    'repetition': 1 if result.get('repetition', '').strip().lower() == 'yes' else 0,
                    'errors': 1 if result.get('errors', '').strip().lower() == 'yes' else 0,
                    'negative': 1 if result.get('negative', '').strip().lower() == 'yes' else 0,
                    'switch': 1 if result.get('switch', '').strip().lower() == 'yes' else 0,
                    'irrelevant': 1 if result.get('irrelevant', '').strip().lower() == 'yes' else 0,
                    'complex': 1 if result.get('complex', '').strip().lower() == 'yes' else 0
                }
            except json.JSONDecodeError as je:
                print(f"JSON parsing error: {je}")
                print(f"Attempted to parse: {json_str}")
                answer_lower = answer.lower()
                return {
                    'repetition': 1 if '"repetition": "yes"' in answer_lower or "'repetition': 'yes'" in answer_lower else 0,
                    'errors': 1 if '"errors": "yes"' in answer_lower or "'errors': 'yes'" in answer_lower else 0,
                    'negative': 1 if '"negative": "yes"' in answer_lower or "'negative': 'yes'" in answer_lower else 0,
                    'switch': 1 if '"switch": "yes"' in answer_lower or "'switch': 'yes'" in answer_lower else 0,
                    'irrelevant': 1 if '"irrelevant": "yes"' in answer_lower or "'irrelevant': 'yes'" in answer_lower else 0,
                    'complex': 1 if '"complex": "yes"' in answer_lower or "'complex': 'yes'" in answer_lower else 0
                }

        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"All {max_retries} attempts failed for conversation: {conversation[:50]}...")
                return {'repetition': 0, 'errors': 0, 'negative': 0, 'switch': 0, 'irrelevant': 0, 'complex': 0}

    return {'repetition': 0, 'errors': 0, 'negative': 0, 'switch': 0, 'irrelevant': 0, 'complex': 0}


tqdm.pandas()

# Apply evaluation and expand results into separate columns
results = df["conversation"].progress_apply(evaluate_conversation)
df["repetition"] = results.apply(lambda x: x['repetition'])
df["errors"] = results.apply(lambda x: x['errors'])
df["negative"] = results.apply(lambda x: x['negative'])
df["switch"] = results.apply(lambda x: x['switch'])
df["irrelevant"] = results.apply(lambda x: x['irrelevant'])
df["complex"] = results.apply(lambda x: x['complex'])

df.to_csv("churn.csv", index=False)
print(f"Repetition detected: {df['repetition'].sum()}/{len(df)}")
print(f"Errors detected: {df['errors'].sum()}/{len(df)}")
print(f"Negative sentiment: {df['negative'].sum()}/{len(df)}")
print(f"Topic switching: {df['switch'].sum()}/{len(df)}")
print(f"Irrelevant content: {df['irrelevant'].sum()}/{len(df)}")
print(f"Complex responses: {df['complex'].sum()}/{len(df)}")