import pandas as pd
import openai
import time
import json
import re
import os
import concurrent.futures
from tqdm import tqdm  # For progress bar display

# Configuration Parameters
CONFIG = {
    "DEEPSEEK_API_KEY": "xxx",
    "INPUT_EXCEL_PATH": "/Users/huangxinjie/Desktop/personality/CFAEFA/survey.xlsx",
    "OUTPUT_EXCEL_PATH": "/Users/huangxinjie/Desktop/personality/CFAEFA/surveydeepseek.xlsx",
    "SHEET_NAME": "Sheet3",
    "CHAT_CONTENT_COLUMN": "merged_messages",
    "RESPONSE_ID_COLUMN": "responseid",
    "MAX_RETRIES": 3,
    "API_TIMEOUT": 60,
    "DEEPSEEK_MODEL": "deepseek-chat",
    "PARALLEL_WORKERS": 8,
    "OUTPUT_BATCH_SIZE": 100,
    "RETRY_DELAY": {
        1: 5,
        2: 10,
        3: 20,
    }
}


def initialize_deepseek_client():
    """Initialize DeepSeek API client with custom base URL"""
    return openai.OpenAI(
        api_key=CONFIG["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1"
    )


def rate_chat_with_deepseek(client, chat_content: str, response_id: str):
    """Rate chat using DeepSeek model with retry strategy"""
    for attempt in range(1, CONFIG["MAX_RETRIES"] + 1):
        try:
            response = client.chat.completions.create(
                model=CONFIG["DEEPSEEK_MODEL"],
                messages=[
                    {
                        "role": "system",
                        "content": f"""
                You are a chat rating expert. Analyze a conversation where a bot shared an embarrassing story about pants splitting while biking and asked the human to share something in return.
                Provide ratings with these specific fields (MUST FOLLOW STRICTLY):

                1. Disclosure (0 or 1, BINARY ONLY):
                   - 1 = Human shared any personal information (experiences, emotions, relationships, demographics, etc)
                   - 0 = No personal information shared
                2. Disclosure Depth (1-7): The extent to which human generally shared personal information. (1=low, 7=high)
                3. Reason:
                   Brief explanation for ratings (MUST reference both Disclosure and Disclosure Depth)
                4. responseID:
                   Include the response ID: {response_id}

                IMPORTANT:
                - Disclosure MUST be 0 or 1 (NO other values)
                - Disclosure Depth MUST be 1-7
                - Return STRICT JSON format with NO extra text:

                {{
                    "Disclosure": 0 or 1,
                    "Disclosure Depth": 1-7,
                    "Reason": "Explanation",
                    "responseID": "{response_id}"
                }}
                    """},
                    {"role": "user", "content": f"Rate this feedback (responseID: {response_id}):\n{chat_content}"}
                ],
                temperature=0.1,
                max_tokens=700,
                timeout=CONFIG["API_TIMEOUT"]
            )
            return response.choices[0].message.content
        except openai.RateLimitError:
            delay = CONFIG["RETRY_DELAY"].get(attempt, 30)
            print(f"⚠️ DeepSeek rate limit (Attempt {attempt}/{CONFIG['MAX_RETRIES']}, ID:{response_id}), waiting {delay}s...")
            time.sleep(delay)
        except Exception as e:
            print(f"❌ DeepSeek API error (Attempt {attempt}/{CONFIG['MAX_RETRIES']}, ID:{response_id}): {str(e)}")
            if attempt < CONFIG["MAX_RETRIES"]:
                time.sleep(5)
    return None


def clean_response_text(text):
    """Clean response text by removing markdown and plugin formats"""
    if not text:
        return text
    text = re.sub(r'^```(json)?\s*', '', text)
    text = re.sub(r'\s*```\s*$', '', text)
    text = re.sub(r'^\{"name":"JSONPlugin","parameters":\{"input":"JSONStringify\[', '', text)
    text = re.sub(r'\]\}"\}\]$', '', text)
    return text.replace('\\"', '"').strip()


def parse_rating_response(response_text: str, response_id: str):
    """Parse rating response with error correction"""
    if not response_text:
        return None, f"Empty response (ID:{response_id})"
    cleaned = clean_response_text(response_text)
    try:
        if not (cleaned.startswith("{") and cleaned.endswith("}")):
            return None, f"Non-JSON format (ID:{response_id}): {cleaned[:30]}..."
        data = json.loads(cleaned)
        required_fields = ["Disclosure", "Disclosure Depth", "Reason", "responseID"]
        for field in required_fields:
            if field not in data:
                data[field] = None if field != "responseID" else response_id
                print(f"⚠️ Missing field '{field}' filled (ID:{response_id})")
        if data["Disclosure"] not in [0, 1]:
            data["Disclosure"] = 1 if data.get("Disclosure Depth", 1) > 1 else 0
            print(f"⚠️ Auto-corrected Disclosure: {data['Disclosure']} (ID:{response_id})")
        if data.get("Disclosure Depth") not in range(1, 8):
            data["Disclosure Depth"] = 2 if data["Disclosure"] == 1 else 1
            print(f"⚠️ Auto-corrected Disclosure Depth: {data['Disclosure Depth']} (ID:{response_id})")
        if str(data["responseID"]) != str(response_id):
            data["responseID"] = response_id
            print(f"⚠️ Corrected responseID: {response_id} (ID:{response_id})")
        return {
            "Disclosure": data["Disclosure"],
            "Disclosure Depth": data["Disclosure Depth"],
            "Reason": data["Reason"],
            "responseID": data["responseID"]
        }, None
    except Exception as e:
        return None, f"Parsing failed (ID:{response_id}): {str(e)}"


def process_row(client, row, save_dir):
    """Process a single row with error handling"""
    response_id = str(row[CONFIG["RESPONSE_ID_COLUMN"]]).strip()
    content = str(row[CONFIG["CHAT_CONTENT_COLUMN"]]).strip()
    if not content:
        print(f"⏭️ Skipping empty content (ID:{response_id})")
        return row.name, {
            "Disclosure": 0,
            "Disclosure Depth": 1,
            "Reason": "Empty conversation, no personal information disclosed",
            "responseID": response_id
        }
    api_response = rate_chat_with_deepseek(client, content, response_id)
    api_file = os.path.join(save_dir, f"api_{response_id}.json")
    with open(api_file, "w", encoding="utf-8") as f:
        f.write(api_response or "null")
    ratings, error = parse_rating_response(api_response, response_id)
    if ratings:
        print(f"✅ Success (ID:{response_id}): Disclosure={ratings['Disclosure']}, Depth={ratings['Disclosure Depth']}")
        return row.name, ratings
    else:
        print(f"❌ Failure (ID:{response_id}): {error}")
        return row.name, {
            "Disclosure": None,
            "Disclosure Depth": None,
            "Reason": error,
            "responseID": response_id
        }


def process_chat_ratings():
    """Main processing function with batch handling"""
    client = initialize_deepseek_client()
    save_dir = "/Users/huangxinjie/Desktop/personality/deepseek_responses"
    os.makedirs(save_dir, exist_ok=True)
    try:
        df = pd.read_excel(CONFIG["INPUT_EXCEL_PATH"], sheet_name=CONFIG["SHEET_NAME"])
    except Exception as e:
        print(f"📖 Excel read error: {str(e)}")
        return
    required_cols = [CONFIG["RESPONSE_ID_COLUMN"], CONFIG["CHAT_CONTENT_COLUMN"]]
    for col in required_cols:
        if col not in df.columns:
            print(f"🚨 Missing column '{col}', process terminated")
            return
    result_cols = ["Disclosure", "Disclosure Depth", "Reason", "responseID"]
    if not all(col in df.columns for col in result_cols):
        df[result_cols] = None
    total_rows = len(df)
    print(f"=== Starting DeepSeek chat rating process ===\n"
          f"Total records: {total_rows}\n"
          f"Model: {CONFIG['DEEPSEEK_MODEL']}\n"
          f"Output path: {CONFIG['OUTPUT_EXCEL_PATH']}\n"
          f"API responses: {save_dir}")
    progress_bar = tqdm(total=total_rows, desc="Processing", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    processed = 0
    for batch_start in range(0, total_rows, CONFIG["OUTPUT_BATCH_SIZE"]):
        batch_end = min(batch_start + CONFIG["OUTPUT_BATCH_SIZE"], total_rows)
        batch_df = df[batch_start:batch_end].copy()
        unprocessed = batch_df[batch_df["Disclosure Depth"].isnull()]
        if len(unprocessed) == 0:
            print(f"⏩ Skipping processed batch {batch_start+1}-{batch_end}")
            processed += len(batch_df)
            progress_bar.update(len(batch_df))
            continue
        print(f"🔧 Processing batch {batch_start+1}-{batch_end} ({len(unprocessed)}/{len(batch_df)} unprocessed)")
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["PARALLEL_WORKERS"]) as executor:
            future_to_index = {
                executor.submit(process_row, client, row, save_dir): row.name
                for _, row in unprocessed.iterrows()
            }
            for future in concurrent.futures.as_completed(future_to_index):
                try:
                    index, result = future.result()
                    df.loc[index, result_cols] = [
                        result["Disclosure"],
                        result["Disclosure Depth"],
                        result["Reason"],
                        result["responseID"]
                    ]
                    processed += 1
                    progress_bar.update(1)
                except Exception as e:
                    print(f"🔥 Unknown error (row {index}): {str(e)}")
                    processed += 1
                    progress_bar.update(1)
        print(f"💾 Saving batch {batch_start+1}-{batch_end}, processed {processed}/{total_rows}")
        df.to_excel(CONFIG["OUTPUT_EXCEL_PATH"], index=False)
    progress_bar.close()
    print(f"\n=== Process completed ===\n"
          f"Processed: {processed}/{total_rows}\n"
          f"Results saved to: {CONFIG['OUTPUT_EXCEL_PATH']}\n"
          f"API responses: {save_dir}\n"
          f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")


if __name__ == "__main__":
    start_time = time.time()
    process_chat_ratings()
