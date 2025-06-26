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
    "OPENAI_API_KEY": "xxx", # put your api key here
    "INPUT_EXCEL_PATH": "/Users/huangxinjie/Desktop/personality/CFAEFA/survey.xlsx",
    "OUTPUT_EXCEL_PATH": "/Users/huangxinjie/Desktop/personality/CFAEFA/surveygpt2.xlsx",
    "SHEET_NAME": "Sheet3",
    "CHAT_CONTENT_COLUMN": "merged_messages",
    "RESPONSE_ID_COLUMN": "responseid",
    "MAX_RETRIES": 5,  # Maximum number of retries
    "API_TIMEOUT": 90,  # API timeout in seconds
    "OPENAI_MODEL": "gpt-4.1",  # GPT model to use
    "PARALLEL_WORKERS": 2,  # Number of parallel worker threads
    "OUTPUT_BATCH_SIZE": 30,  # Batch size for saving outputs
    "RETRY_DELAY": {  # Graded retry strategy
        1: 5,  # 5-second delay for 1st retry
        2: 15,  # 15-second delay for 2nd retry
        3: 30,  # 30-second delay for 3rd retry
        4: 60,  # 60-second delay for 4th retry
    }
}


def initialize_openai_client():
    """Initialize OpenAI API client"""
    return openai.OpenAI(
        api_key=CONFIG["OPENAI_API_KEY"],
    )


def rate_chat_with_gpt(client, chat_content: str, response_id: str):
    """Rate the chat using GPT with optimized Disclosure Depth terminology"""
    for attempt in range(1, CONFIG["MAX_RETRIES"] + 1):
        try:
            response = client.chat.completions.create(
                model=CONFIG["OPENAI_MODEL"],
                messages=[
                    {"role": "system", "content": f"""
You are a chat rating expert. Your task is to analyze a conversation where a bot shared an embarrassing story about pants splitting while biking and asked the human to share something in return.
Please provide ratings with these specific fields:
1. Disclosure (1/0): Whether human shared personal information in general during the chat (e.g., their name, personal details like they are married, a story about themselves) on the binary scale (1 = yes, 0 = no)
2. Disclosure Depth (1-7): The extent to which human generally shared personal information. (1=low, 7=high)
3. Reason: Provide a brief explanation for the ratings you've given
4. Response_id: Include the response ID of this chat

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
                temperature=0.1,  # Reduce randomness for consistent results
                max_tokens=700,
                timeout=CONFIG["API_TIMEOUT"]
            )
            return response.choices[0].message.content
        except openai.RateLimitError:
            delay = CONFIG["RETRY_DELAY"].get(attempt, 120)
            print(f"⚠️ Rate limit (Attempt {attempt}/{CONFIG['MAX_RETRIES']}, ID:{response_id}), waiting {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            print(f"❌ API error (Attempt {attempt}/{CONFIG['MAX_RETRIES']}, ID:{response_id}): {str(e)}")
            if attempt < CONFIG["MAX_RETRIES"]:
                time.sleep(5)
    return None


def clean_response_text(text):
    """Deep clean response text to remove invalid formats"""
    if not text:
        return text
    # Remove Markdown code block markers
    text = re.sub(r'^```(json)?\s*', '', text)
    text = re.sub(r'\s*```\s*$', '', text)
    # Remove plugin call formats
    text = re.sub(r'^\{"name":"JSONPlugin","parameters":\{"input":"JSONStringify\[', '', text)
    text = re.sub(r'\]\}"\}\]$', '', text)
    # Handle escaped characters
    return text.replace('\\"', '"').strip()


def parse_rating_response(response_text: str, response_id: str):
    """Intelligently parse rating responses to adapt to Disclosure Depth terminology"""
    if not response_text:
        return None, f"Empty response (ID:{response_id})"

    cleaned = clean_response_text(response_text)
    try:
        if not (cleaned.startswith("{") and cleaned.endswith("}")):
            return None, f"Non-JSON format (ID:{response_id}): {cleaned[:30]}..."

        data = json.loads(cleaned)
        required_fields = ["Disclosure", "Disclosure Depth", "Reason", "responseID"]

        # 1. Handle missing fields
        for field in required_fields:
            if field not in data:
                data[field] = None if field != "responseID" else response_id
                print(f"⚠️ Missing field '{field}' filled (ID:{response_id})")

        # 2. Auto-correct Disclosure values
        if data["Disclosure"] not in [0, 1]:
            if data["Disclosure Depth"] is not None:
                data["Disclosure"] = 1 if data["Disclosure Depth"] > 1 else 0
                print(f"⚠️ Disclosure auto-corrected to: {data['Disclosure']} (ID:{response_id})")
            else:
                data["Disclosure"] = 0
                print(f"⚠️ Unknown Disclosure, set to 0 (ID:{response_id})")

        # 3. Auto-correct Disclosure Depth values
        if data["Disclosure Depth"] is not None and not (1 <= data["Disclosure Depth"] <= 7):
            if data["Disclosure"] == 1:
                data["Disclosure Depth"] = 2  # Default to basic preference
            else:
                data["Disclosure Depth"] = 1  # No personal information
            print(f"⚠️ Disclosure Depth auto-corrected to: {data['Disclosure Depth']} (ID:{response_id})")

        # 4. Correct responseID consistency
        if str(data["responseID"]) != str(response_id):
            data["responseID"] = response_id
            print(f"⚠️ responseID corrected to: {response_id} (ID:{response_id})")

        return {
            "Disclosure": data["Disclosure"],
            "Disclosure Depth": data["Disclosure Depth"],
            "Reason": data["Reason"],
            "responseID": data["responseID"]
        }, None

    except Exception as e:
        return None, f"Parsing failed (ID:{response_id}): {str(e)}"


def process_row(client, row, save_dir):
    """Process a single row of data, updating field name references"""
    response_id = str(row[CONFIG["RESPONSE_ID_COLUMN"]]).strip()
    content = str(row[CONFIG["CHAT_CONTENT_COLUMN"]]).strip()

    # Preprocess empty content
    if not content:
        print(f"⏭️ Skipping empty content (ID:{response_id})")
        return row.name, {
            "Disclosure": 0,
            "Disclosure Depth": 1,
            "Reason": "Empty conversation, no personal information disclosed",
            "responseID": response_id
        }

    # Call API
    api_response = rate_chat_with_gpt(client, content, response_id)

    # Save raw API response
    api_file = os.path.join(save_dir, f"api_{response_id}.json")
    with open(api_file, "w", encoding="utf-8") as f:
        f.write(api_response or "null")

    # Parse response
    ratings, error = parse_rating_response(api_response, response_id)
    if ratings:
        print(f"✅ Processing successful (ID:{response_id}): Disclosure={ratings['Disclosure']}, Depth={ratings['Disclosure Depth']}")
        return row.name, ratings
    else:
        print(f"❌ Processing failed (ID:{response_id}): {error}")
        return row.name, {
            "Disclosure": None,
            "Disclosure Depth": None,
            "Reason": error,
            "responseID": response_id
        }


def process_chat_ratings():
    """Main processing function, updating field name references"""
    client = initialize_openai_client()
    save_dir = "/Users/huangxinjie/Desktop/personality/gpt4_responses"
    os.makedirs(save_dir, exist_ok=True)

    try:
        df = pd.read_excel(CONFIG["INPUT_EXCEL_PATH"], sheet_name=CONFIG["SHEET_NAME"])
    except Exception as e:
        print(f"📖 Excel read failed: {str(e)}")
        return

    # Validate required columns
    required_cols = [CONFIG["RESPONSE_ID_COLUMN"], CONFIG["CHAT_CONTENT_COLUMN"]]
    for col in required_cols:
        if col not in df.columns:
            print(f"🚨 Missing required column '{col}', processing terminated")
            return

    # Create result columns
    result_cols = ["Disclosure", "Disclosure Depth", "Reason", "responseID"]
    if not all(col in df.columns for col in result_cols):
        df[result_cols] = None

    total_rows = len(df)
    print(f"=== Starting chat rating processing ===\n"
          f"Total records: {total_rows}\n"
          f"Model: {CONFIG['OPENAI_MODEL']}\n"
          f"Save path: {CONFIG['OUTPUT_EXCEL_PATH']}\n"
          f"API responses save: {save_dir}")

    progress_bar = tqdm(total=total_rows, desc="Processing progress",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    processed = 0

    # Batch processing mechanism (supports resume on break)
    for batch_start in range(0, total_rows, CONFIG["OUTPUT_BATCH_SIZE"]):
        batch_end = min(batch_start + CONFIG["OUTPUT_BATCH_SIZE"], total_rows)
        batch_df = df[batch_start:batch_end].copy()

        # Filter processed records (judge by Disclosure Depth being None)
        unprocessed = batch_df[batch_df["Disclosure Depth"].isnull()]
        if len(unprocessed) == 0:
            print(f"⏩ Skipping processed batch {batch_start+1}-{batch_end}")
            processed += len(batch_df)
            progress_bar.update(len(batch_df))
            continue

        print(f"🔧 Processing batch {batch_start+1}-{batch_end} ({len(unprocessed)}/{len(batch_df)} unprocessed records)")
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

        # Save intermediate results
        print(f"💾 Saving batch {batch_start+1}-{batch_end}, processed {processed}/{total_rows}")
        df.to_excel(CONFIG["OUTPUT_EXCEL_PATH"], index=False)

    progress_bar.close()
    print(f"\n=== Processing completed ===\n"
          f"Successfully processed: {processed} records\n"
          f"Results saved to: {CONFIG['OUTPUT_EXCEL_PATH']}\n"
          f"API responses saved to: {save_dir}\n"
          f"Processing time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")


if __name__ == "__main__":
    start_time = time.time()
    process_chat_ratings()
