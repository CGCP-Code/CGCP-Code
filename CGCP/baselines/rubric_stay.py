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
        "1. Does the user thank or compliment the chatbot for its help, quality, performance, or abilities?\n"
        "2. Does the user engage in a diverse and lengthy conversation with the chatbot, covering multiple topics or domains?\n"
        "3. Does the user ask follow-up questions or request more information from the chatbot that show curiosity and interest in learning more?\n"
        "4. Does the user accept or agree with the AI agent's suggestions, recommendations, or advice?\n"
        "5. Does the user enjoy, appreciate, and learn from response provided by the chatbot?\n"
        "6. If the user does not express any frustration, confusion, or dissatisfaction with the chatbot’s responses or queries throughout the conversation?\n\n"
        "IMPORTANT: Ignore any entries marked as 'EMPTY'. Only evaluate actual questions and responses.\n\n"
        f"Conversation:\n{conversation}\n\n"
        "Respond ONLY with a JSON object in this exact format:\n"
        '{"gratitude": "Yes", "engagement": "No", "followup": "Yes", "acceptance": "No", "learning": "Yes", "noFrustration": "No"}\n\n'
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
                    'gratitude': 1 if result.get('gratitude', '').strip().lower() == 'yes' else 0,
                    'engagement': 1 if result.get('engagement', '').strip().lower() == 'yes' else 0,
                    'followup': 1 if result.get('followup', '').strip().lower() == 'yes' else 0,
                    'acceptance': 1 if result.get('acceptance', '').strip().lower() == 'yes' else 0,
                    'learning': 1 if result.get('learning', '').strip().lower() == 'yes' else 0,
                    'noFrustration': 1 if result.get('noFrustration', '').strip().lower() == 'yes' else 0
                }
            except json.JSONDecodeError as je:
                print(f"JSON parsing error: {je}")
                print(f"Attempted to parse: {json_str}")
                answer_lower = answer.lower()
                return {
                    'gratitude': 1 if '"gratitude": "yes"' in answer_lower or "'gratitude': 'yes'" in answer_lower else 0,
                    'engagement': 1 if '"engagement": "yes"' in answer_lower or "'engagement': 'yes'" in answer_lower else 0,
                    'followup': 1 if '"followup": "yes"' in answer_lower or "'followup': 'yes'" in answer_lower else 0,
                    'acceptance': 1 if '"acceptance": "yes"' in answer_lower or "'acceptance': 'yes'" in answer_lower else 0,
                    'learning': 1 if '"learning": "yes"' in answer_lower or "'learning': 'yes'" in answer_lower else 0,
                    'noFrustration': 1 if '"nofrustration": "yes"' in answer_lower or "'nofrustration': 'yes'" in answer_lower else 0
                }

        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"All {max_retries} attempts failed for conversation: {conversation[:50]}...")
                return {'gratitude': 0, 'engagement': 0, 'followup': 0, 'acceptance': 0, 'learning': 0,
                        'noFrustration': 0}

    return {'gratitude': 0, 'engagement': 0, 'followup': 0, 'acceptance': 0, 'learning': 0, 'noFrustration': 0}


tqdm.pandas()

# Apply evaluation and expand results into separate columns
results = df["conversation"].progress_apply(evaluate_conversation)
df["gratitude"] = results.apply(lambda x: x['gratitude'])
df["engagement"] = results.apply(lambda x: x['engagement'])
df["followup"] = results.apply(lambda x: x['followup'])
df["acceptance"] = results.apply(lambda x: x['acceptance'])
df["learning"] = results.apply(lambda x: x['learning'])
df["noFrustration"] = results.apply(lambda x: x['noFrustration'])

df.to_csv("stay.csv", index=False)
print(f"Gratitude expressed: {df['gratitude'].sum()}/{len(df)}")
print(f"Meaningful engagement: {df['engagement'].sum()}/{len(df)}")
print(f"Follow-up questions: {df['followup'].sum()}/{len(df)}")
print(f"Acceptance of suggestions: {df['acceptance'].sum()}/{len(df)}")
print(f"Learning demonstrated: {df['learning'].sum()}/{len(df)}")
print(f"No frustration shown: {df['noFrustration'].sum()}/{len(df)}")