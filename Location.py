import json
import requests
import subprocess
import pandas as pd
import threading
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery
import sys
import os

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1, encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# Create log file immediately
logger = Logger("pipeline.log")
sys.stdout = logger

# ============================================================
# CONFIGURATION
# ============================================================
GCP_PROJ = "aif-usr-p-ep-cg-wcdm-0767"
BQ_DATASET = "aif_wcdm_osm_pipeline"
BQ_LOCATION = "US"

TEST_PAIRS_FILE = "test_pairs.json"
SUMMARY_OUTPUT_FILE = "pipeline_selected_summary.json"

# ============================================================
# SOURCE TABLES
# ============================================================
ACTIVITY_TABLE = f"{GCP_PROJ}.{BQ_DATASET}.SCENARIO_6_ACTIVITY_CODE_DETAILS_ARRAY"
MEMBER_TABLE = f"{GCP_PROJ}.{BQ_DATASET}.SCENARIO_6_MEMBER"
MAP_TABLE = f"{GCP_PROJ}.{BQ_DATASET}.SCENARIO_6_MAP"
CLUSTER_TABLE = f"{GCP_PROJ}.{BQ_DATASET}.SCENARIO_6_CLUSTER"
ALL_RESULTS_TABLE = f"{GCP_PROJ}.{BQ_DATASET}.all_results"

# ============================================================
# APIS
# ============================================================
CLUSTER_SELECTION_URL = (
    "https://cluster-selection-api-68004442910.us-central1.run.app"
    "/cluster-selection/v2/cui_matching"
)
TRANSACTION_SELECTION_URL = (
    "https://transaction-selection-api-705290722717.us-central1.run.app"
    "/transaction_selection"
)
CLUSTER_SET = "loinc_document_v001"
TOP_K = 10
TOP_P = 20
COMBINED = True
GEMINI_MODEL = "gemini-2.5-flash"

# ============================================================
# PARALLELISM / BATCHING
# ============================================================
TRANSACTION_BATCH_SIZE = 25
TRANSACTION_MAX_WORKERS = 30
DOCUMENT_TRANSACTION_BATCH_SIZE = 25
DOCUMENT_TRANSACTION_MAX_WORKERS = 30
CLASS_TRANSACTION_BATCH_SIZE = 25
CLASS_TRANSACTION_MAX_WORKERS = 30

# ============================================================
# GCLOUD
# ============================================================
def resolve_gcloud_path():
    candidates = []
    gcloud_in_path = shutil.which("gcloud")
    if gcloud_in_path:
        candidates.append(gcloud_in_path)

    candidates.extend([
        r"C:\Users\M271333\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
        
    ])

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        normalized = os.path.normcase(os.path.normpath(candidate))
        if normalized not in seen:
            seen.add(normalized)
            if os.path.exists(candidate):
                return candidate

    if gcloud_in_path:
        return gcloud_in_path

    raise FileNotFoundError(
        "Google Cloud SDK gcloud executable was not found. "
        "Install or configure gcloud on this machine."
    )

GCLOUD_LOC = resolve_gcloud_path()
headers = None
print_lock = threading.Lock()

# ============================================================
# AUTHENTICATION
# ============================================================
def gcp_update_header():
    global headers
    tmp = subprocess.run(
        [GCLOUD_LOC, "auth", "print-identity-token"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if tmp.returncode != 0:
        raise Exception(
            f"Cannot get GCP identity token.\nSTDERR: {tmp.stderr}"
        )
    identity_token = tmp.stdout.strip()
    if not identity_token:
        raise Exception("GCloud returned an empty identity token.")
    headers = {
        "Authorization": f"Bearer {identity_token}",
        "Content-Type": "application/json",
    }
    print("GCP identity token obtained successfully.")

# ============================================================
# BIGQUERY
# ============================================================
def get_bq_client():
    return bigquery.Client(project=GCP_PROJ, location=BQ_LOCATION)

# ============================================================
# CLUSTER SELECTION
# ============================================================
def call_cluster_selection(search_document_name, search_context=None):
    context_text = search_context if (search_context and search_context.strip()) else search_document_name
    payload = {
        "cluster_set": CLUSTER_SET,
        "text_list": [search_document_name],
        "context_list": [context_text],
        "top_k": TOP_K,
        "top_p": TOP_P,
        "combined": COMBINED,
    }
    response = requests.post(
        CLUSTER_SELECTION_URL,
        json=payload,
        headers=headers,
        timeout=120,
    )
    if response.status_code != 200:
        raise Exception(
            f"Cluster Selection failed.\nStatus: {response.status_code}\nResponse: {response.text}"
        )
    result = response.json()
    with open("cluster_selection_response.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result

def extract_member_names(obj):
    member_names = set()
    def recursive_extract(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "member_name":
                    if isinstance(child, str):
                        name = child.strip()
                        if name:
                            member_names.add(name)
                    elif isinstance(child, list):
                        for item in child:
                            if isinstance(item, str):
                                name = item.strip()
                                if name:
                                    member_names.add(name)
                recursive_extract(child)
        elif isinstance(value, list):
            for item in value:
                recursive_extract(item)
    recursive_extract(obj)
    return sorted(member_names)

def get_probable_documents(search_document_name, search_context=None):
    cluster_response = call_cluster_selection(search_document_name, search_context)
    probable_documents = extract_member_names(cluster_response)
    print("\n========================================")
    print("PROBABLE DOCUMENT NAMES")
    print("========================================")
    for index, document_name in enumerate(probable_documents, start=1):
        print(f"{index}. {document_name}")
    print(f"Total probable documents: {len(probable_documents)}")
    return probable_documents

# ============================================================
# ACTIVITY DISCOVERY
# ============================================================
def get_activities_for_documents(document_names):
    if not document_names:
        return pd.DataFrame()
    client = get_bq_client()
    query = f"""
        SELECT
            document_name,
            activity_details.activity_id AS activity_id,
            activity_details.activity_name AS activity_name,
            activity_details.activity_definition AS activity_definition
        FROM `{ACTIVITY_TABLE}`
        CROSS JOIN UNNEST(activity_details) AS activity_details
        WHERE document_name IN UNNEST(@document_names)
        ORDER BY document_name, activity_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("document_names", "STRING", document_names)
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    if df.empty:
        return df
    df = df.drop_duplicates(
        subset=[
            "document_name",
            "activity_id",
            "activity_name",
            "activity_definition",
        ]
    ).reset_index(drop=True)
    df = df[df["activity_name"].notna()].copy()
    df = df[df["activity_name"].astype(str).str.strip() != ""].copy()
    df.reset_index(drop=True, inplace=True)
    return df

# ============================================================
# STAGE 1: ACTIVITY TRANSACTION SELECTION
# ============================================================
def generate_activity_transaction_payload(df, search_document_name, search_context=None):
    has_context = bool(search_context and search_context.strip())
    transactions = []
    
    for _, row in df.iterrows():
        if has_context:
            transactions.append([
                search_document_name,
                search_context.strip(),
                row["activity_name"],
                row["activity_definition"],
            ])
        else:
            transactions.append([
                search_document_name,
                row["activity_name"],
                row["activity_definition"],
            ])

    if has_context:
        instructions = [
            "Each transaction contains exactly 4 elements.",
            "The first element is the search document name.",
            "The second element is the search context.",
            "The third element is the candidate activity name.",
            "The fourth element is the candidate activity definition.",
        ]
        elements = {
            "search_document_name": (
                "The original document name being searched and the target "
                "against which the candidate activity is evaluated."
            ),
            "search_context": (
                "The specific user or clinical context provided to qualify "
                "and guide the relevance evaluation."
            ),
            "activity_name": "The candidate activity name.",
            "activity_definition": "The definition of the candidate activity.",
        }
        guidelines = [
            "Determine whether the candidate activity is related to the search document name in the given search context.",
            "Consider synonyms and synonymous terminology.",
            "Consider procedural relationships between the document and activity.",
            "Consider the clinical context provided by both the document and the explicit search context.",
            "Consider parental and child classifications.",
            "Do not require an exact lexical match.",
            "A semantic equivalent or closely related activity under the provided context should be considered related.",
            "Use Activity Name first and Activity Definition to resolve ambiguity.",
        ]
    else:
        instructions = [
            "Each transaction contains exactly 3 elements.",
            "The first element is the search document name.",
            "The second element is the candidate activity name.",
            "The third element is the candidate activity definition.",
        ]
        elements = {
            "search_document_name": (
                "The original document name being searched and the target "
                "against which the candidate activity is evaluated."
            ),
            "activity_name": "The candidate activity name.",
            "activity_definition": "The definition of the candidate activity.",
        }
        guidelines = [
            "Determine whether the candidate activity is related to the search document name.",
            "Consider synonyms and synonymous terminology.",
            "Consider procedural relationships between the document and activity.",
            "Consider the clinical context of the document and activity.",
            "Consider parental and child classifications.",
            "Do not require an exact lexical match.",
            "A semantic equivalent or closely related activity should be considered related.",
            "Use Activity Name first and Activity Definition to resolve ambiguity.",
        ]

    return {
        "model_name": GEMINI_MODEL,
        "transactions": transactions,
        "objective": (
            "For each transaction, determine whether the candidate activity "
            "is related to the search document name" +
            (" within the specified search context." if has_context else ".") +
            " The purpose is to retain activities that are relevant to the searched document."
        ),
        "definitions": {
            "instructions": instructions,
            "elements": elements,
        },
        "analysis_guidelines": guidelines,
        "output_spec": {
            "instructions": [
                "Return exactly one result for every transaction.",
                "Return results in exactly the same order as the input transactions.",
                "Return Yes when the candidate activity is related to the search document.",
                "Return No when the candidate activity is not related to the search document.",
                "Provide a short reasoning explaining the decision.",
            ],
            "response_fields": {
                "answer": {
                    "Yes": "The candidate activity is related to the search document.",
                    "No": "The candidate activity is not related to the search document.",
                },
                "reasoning": "Brief explanation of the decision.",
            },
        },
    }

def send_transaction_selection(payload, max_retries=5, retry_delay=3):
    """Sends transactions to the selection API.

    Automatically retries if the LLM produces a mismatched count (400 error)
    or if a transient network/server error occurs.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                TRANSACTION_SELECTION_URL,
                json=payload,
                headers=headers,
                timeout=300,
            )

            if response.status_code == 200:
                response_json = response.json()
                if "output" not in response_json:
                    raise Exception(
                        f"Transaction Selection response does not contain 'output'.\nResponse: {response_json}"
                    )
                return response_json["output"]

            # Handle model hallucination on item count (e.g. expected 50, got 51)
            if response.status_code == 400 and "incorrect number of items" in response.text:
                with print_lock:
                    print(
                        f"[RETRY] Model returned incorrect item count (Attempt {attempt}/{max_retries}). "
                        f"Retrying exact batch in {retry_delay}s..."
                    )
                time.sleep(retry_delay)
                continue

            # Handle rate limits or temporary server errors
            if response.status_code in [429, 500, 502, 503, 504]:
                wait_time = retry_delay * attempt
                with print_lock:
                    print(
                        f"[RETRY] Server returned {response.status_code} (Attempt {attempt}/{max_retries}). "
                        f"Retrying in {wait_time}s..."
                    )
                time.sleep(wait_time)
                continue

            # Non-retryable error
            raise Exception(
                f"Transaction Selection failed.\nStatus: {response.status_code}\nResponse: {response.text}"
            )

        except requests.RequestException as e:
            with print_lock:
                print(f"[RETRY] Request exception on attempt {attempt}/{max_retries}: {str(e)}")
            if attempt == max_retries:
                raise
            time.sleep(retry_delay * attempt)

    raise Exception(f"Transaction Selection failed after {max_retries} attempts.")

def process_activity_batch(batch_df, search_document_name, search_context=None):
    payload = generate_activity_transaction_payload(batch_df, search_document_name, search_context)
    results = send_transaction_selection(payload)
    if not isinstance(results, list):
        raise Exception("Expected Transaction Selection output to be a list.")
    if len(results) != len(batch_df):
        raise Exception(
            "Number of Transaction Selection results does not match input transactions. "
            f"Input: {len(batch_df)}, Output: {len(results)}"
        )
    response_df = pd.DataFrame(results)
    final_df = pd.concat(
        [batch_df.reset_index(drop=True), response_df.reset_index(drop=True)],
        axis=1,
    )
    final_df.insert(0, "search_document_name", search_document_name)
    if search_context:
        final_df.insert(1, "search_context", search_context)
    return final_df

def process_activity_batch_worker(batch_number, start, end, batch_df, search_document_name, search_context=None):
    try:
        with print_lock:
            print(f"[ACTIVITY WORKER] Batch {batch_number}: activities {start + 1}-{end}")
        result_df = process_activity_batch(batch_df, search_document_name, search_context)
        return {"batch_number": batch_number, "data": result_df, "error": None}
    except Exception as e:
        with print_lock:
            print(f"[ACTIVITY WORKER ERROR] Batch {batch_number}: {str(e)}")
        return {"batch_number": batch_number, "data": None, "error": str(e)}

def run_activity_transaction_selection(activities_df, search_document_name, search_context=None):
    if activities_df.empty:
        return pd.DataFrame()
    batches = []
    total = len(activities_df)
    for start in range(0, total, TRANSACTION_BATCH_SIZE):
        end = min(start + TRANSACTION_BATCH_SIZE, total)
        batches.append((len(batches) + 1, start, end, activities_df.iloc[start:end].copy()))
    results = []
    successful = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=TRANSACTION_MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_activity_batch_worker,
                batch_number,
                start,
                end,
                batch_df,
                search_document_name,
                search_context,
            ): batch_number
            for batch_number, start, end, batch_df in batches
        }
        for future in as_completed(futures):
            result = future.result()
            if result["error"]:
                failed += 1
            else:
                successful += 1
                results.append(result["data"])
    print(f"Activity transaction batches: {len(batches)}")
    print(f"Successful: {successful}, Failed: {failed}")
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)

# ============================================================
# CANDIDATE RELATED DOCUMENT NAMES
# ============================================================
def get_related_document_names(transaction_results_df, search_document_name, search_context=None):
    if transaction_results_df.empty:
        return pd.DataFrame()
    yes_df = transaction_results_df[
        transaction_results_df["answer"].astype(str).str.strip().str.lower() == "yes"
    ].copy()
    if yes_df.empty:
        return pd.DataFrame()
    activity_ids = (
        yes_df["activity_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    activity_ids = activity_ids[activity_ids != ""].drop_duplicates().tolist()
    if not activity_ids:
        return pd.DataFrame()
    client = get_bq_client()
    query = f"""
        SELECT DISTINCT
            document_name,
            activity_details.activity_id AS activity_id,
            activity_details.activity_name AS activity_name,
            activity_details.activity_definition AS activity_definition
        FROM `{ACTIVITY_TABLE}`
        CROSS JOIN UNNEST(activity_details) AS activity_details
        WHERE activity_details.activity_id IN UNNEST(@activity_ids)
        ORDER BY document_name
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("activity_ids", "STRING", activity_ids)
        ]
    )
    related = client.query(query, job_config=job_config).to_dataframe()
    if related.empty:
        return related
    related = related.drop_duplicates().reset_index(drop=True)
    related = related[related["document_name"].notna()].copy()
    related["document_name"] = related["document_name"].astype(str).str.strip()
    related = related[related["document_name"] != ""].copy()
    related = related.drop_duplicates(subset=["document_name"]).reset_index(drop=True)
    related.insert(0, "search_document_name", search_document_name)
    if search_context:
        related.insert(1, "search_context", search_context)
    return related

# ============================================================
# STAGE 2: DOCUMENT-NAME TRANSACTION SELECTION
# ============================================================
def generate_document_transaction_payload(document_names, search_document_name, search_context=None):
    has_context = bool(search_context and search_context.strip())
    if has_context:
        transactions = [
            [search_document_name, search_context.strip(), doc_name]
            for doc_name in document_names
        ]
        instructions = [
            "Each transaction contains exactly 3 elements.",
            "The first element is the search document name.",
            "The second element is the search context.",
            "The third element is the candidate document name.",
        ]
        elements = {
            "search_document_name": "The original target document name.",
            "search_context": "The specific context guiding the evaluation.",
            "candidate_document_name": "A candidate related document name.",
        }
        guidelines = [
            "Determine whether the candidate document is related to the search document within the provided context.",
            "Consider synonyms and synonymous terminology.",
            "Consider parent-child and hierarchical relationships.",
            "Consider clinically meaningful associations between concepts under the given context.",
            "Consider whether the candidate represents a procedure, finding, condition, measurement, drug, or other concept that is meaningfully related to the target and context.",
            "Do not require an exact lexical match.",
            "Do not infer a relationship merely because the terms share a word.",
            "Return Yes only when there is a reasonable semantic relationship.",
        ]
    else:
        transactions = [
            [search_document_name, doc_name]
            for doc_name in document_names
        ]
        instructions = [
            "Each transaction contains exactly 2 elements.",
            "The first element is the search document name.",
            "The second element is the candidate document name.",
        ]
        elements = {
            "search_document_name": "The original target document name.",
            "candidate_document_name": "A candidate related document name.",
        }
        guidelines = [
            "Determine whether the candidate document is related to the search document.",
            "Consider synonyms and synonymous terminology.",
            "Consider parent-child and hierarchical relationships.",
            "Consider clinically meaningful associations between concepts.",
            "Consider whether the candidate represents a procedure, finding, condition, measurement, drug, or other concept that is meaningfully related to the target.",
            "Do not require an exact lexical match.",
            "Do not infer a relationship merely because the terms share a word.",
            "Return Yes only when there is a reasonable semantic relationship.",
        ]

    return {
        "model_name": GEMINI_MODEL,
        "transactions": transactions,
        "objective": (
            "Determine whether each candidate document name is related to "
            "the search document name" +
            (" within the specified search context." if has_context else ".") +
            " The candidates are possible related documents discovered from activities. Do not require exact wording."
        ),
        "definitions": {
            "instructions": instructions,
            "elements": elements,
        },
        "analysis_guidelines": guidelines,
        "output_spec": {
            "instructions": [
                "Return exactly one result for every transaction.",
                "Return results in exactly the same order as the input transactions.",
                "Return Yes if the candidate document is related to the search document.",
                "Return No if it is not related.",
                "Provide a short reasoning.",
            ],
            "response_fields": {
                "answer": {
                    "Yes": "The candidate document is related to the search document.",
                    "No": "The candidate document is not related to the search document.",
                },
                "reasoning": "Brief explanation of the relationship decision.",
            },
        },
    }

def process_document_batch(batch_df, search_document_name, search_context=None):
    document_names = batch_df["document_name"].tolist()
    payload = generate_document_transaction_payload(document_names, search_document_name, search_context)
    results = send_transaction_selection(payload)
    if not isinstance(results, list):
        raise Exception("Expected document transaction-selection output to be a list.")
    if len(results) != len(batch_df):
        raise Exception(
            "Document transaction-selection result count mismatch. "
            f"Input: {len(batch_df)}, Output: {len(results)}"
        )
    response_df = pd.DataFrame(results)
    result_df = pd.concat(
        [batch_df.reset_index(drop=True), response_df.reset_index(drop=True)],
        axis=1,
    )
    return result_df

def process_document_batch_worker(batch_number, batch_df, search_document_name, search_context=None):
    try:
        with print_lock:
            print(f"[DOCUMENT WORKER] Batch {batch_number}: {len(batch_df)} documents")
        result_df = process_document_batch(batch_df, search_document_name, search_context)
        return {"batch_number": batch_number, "data": result_df, "error": None}
    except Exception as e:
        with print_lock:
            print(f"[DOCUMENT WORKER ERROR] Batch {batch_number}: {str(e)}")
        return {"batch_number": batch_number, "data": None, "error": str(e)}

def run_document_transaction_selection(related_documents_df, search_document_name, search_context=None):
    if related_documents_df.empty:
        return pd.DataFrame()
    batches = []
    total = len(related_documents_df)
    for start in range(0, total, DOCUMENT_TRANSACTION_BATCH_SIZE):
        end = min(start + DOCUMENT_TRANSACTION_BATCH_SIZE, total)
        batches.append((
            len(batches) + 1,
            related_documents_df.iloc[start:end].copy(),
        ))
    results = []
    successful = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=DOCUMENT_TRANSACTION_MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_document_batch_worker,
                batch_number,
                batch_df,
                search_document_name,
                search_context,
            ): batch_number
            for batch_number, batch_df in batches
        }
        for future in as_completed(futures):
            result = future.result()
            if result["error"]:
                failed += 1
            else:
                successful += 1
                results.append(result["data"])
    print(f"Document-name transaction batches: {len(batches)}")
    print(f"Successful: {successful}, Failed: {failed}")
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)

# ============================================================
# DOCUMENT-CLASS DISCOVERY
# ============================================================
def get_document_classes(document_names):
    if not document_names:
        return pd.DataFrame()
    client = get_bq_client()
    query = f"""
        SELECT DISTINCT
            document_type AS document_name,
            document_class,
            node_id
        FROM `{ALL_RESULTS_TABLE}`
        WHERE document_type IN UNNEST(@document_names)
            AND document_class IS NOT NULL
            AND TRIM(document_class) != ''
        ORDER BY document_name, document_class, node_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("document_names", "STRING", document_names)
        ]
    )
    classes_df = client.query(query, job_config=job_config).to_dataframe()
    return classes_df[["document_name", "document_class", "node_id"]]

# ============================================================
# STAGE 3: DOCUMENT-CLASS TRANSACTION SELECTION
# ============================================================
def generate_class_transaction_payload(class_df, search_document_name, search_context=None):
    has_context = bool(search_context and search_context.strip())
    if has_context:
        transactions = [
            [
                search_document_name,
                search_context.strip(),
                row["document_name"],
                row["document_class"],
            ]
            for _, row in class_df.iterrows()
        ]
        instructions = [
            "Each transaction contains exactly 4 elements.",
            "The first element is the original search document name.",
            "The second element is the search context.",
            "The third element is the candidate related document name.",
            "The fourth element is the candidate document class.",
        ]
        elements = {
            "search_document_name": "The original target document.",
            "search_context": "The specific context provided to guide the evaluation.",
            "candidate_document_name": "A candidate document discovered through previous stages.",
            "document_class": "The candidate document class to evaluate.",
        }
        guidelines = [
            "Determine whether the candidate document class is related to the search document under the given context.",
            "Use the candidate document name as supporting context for the document class.",
            "Evaluate the meaning of the document class within the provided context.",
            "Consider synonyms and synonymous terminology.",
            "Consider parent-child and hierarchical relationships.",
            "Consider clinically meaningful associations in the given context.",
            "Do not require exact lexical overlap.",
            "Do not reject a relationship solely because terminology differs.",
            "Do not mark Yes merely because the document class belongs to the same broad general domain.",
            "Return Yes only when the document class has a reasonable relationship to the search document within the context.",
        ]
    else:
        transactions = [
            [search_document_name, row["document_name"], row["document_class"]]
            for _, row in class_df.iterrows()
        ]
        instructions = [
            "Each transaction contains exactly 3 elements.",
            "The first element is the original search document name.",
            "The second element is the candidate related document name.",
            "The third element is the candidate document class.",
        ]
        elements = {
            "search_document_name": "The original target document.",
            "candidate_document_name": "A candidate document discovered through previous stages.",
            "document_class": "The candidate document class to evaluate.",
        }
        guidelines = [
            "Determine whether the candidate document class is related to the search document.",
            "Use the candidate document name as supporting context for the document class.",
            "Evaluate the meaning of the document class.",
            "Consider synonyms and synonymous terminology.",
            "Consider parent-child and hierarchical relationships.",
            "Consider clinically meaningful associations.",
            "Do not require exact lexical overlap.",
            "Do not reject a relationship solely because terminology differs.",
            "Do not mark Yes merely because the document class is broadly clinical or belongs to the same very general domain.",
            "Return Yes only when the document class has a reasonable relationship to the search document.",
        ]

    return {
        "model_name": GEMINI_MODEL,
        "transactions": transactions,
        "objective": (
            "Determine whether the candidate document class is related to "
            "the search document name" +
            (" within the specified search context." if has_context else ".") +
            " The search document remains the original target for every transaction."
        ),
        "definitions": {
            "instructions": instructions,
            "elements": elements,
        },
        "analysis_guidelines": guidelines,
        "output_spec": {
            "instructions": [
                "Return exactly one result for every transaction.",
                "Return results in exactly the same order as the input transactions.",
                "Return Yes when the candidate document class is related to the search document.",
                "Return No when the candidate document class is not related.",
                "Provide a short reasoning explaining the decision.",
            ],
            "response_fields": {
                "answer": {
                    "Yes": "The candidate document class is related to the search document.",
                    "No": "The candidate document class is not related to the search document.",
                },
                "reasoning": "Brief explanation of the decision.",
            },
        },
    }

def select_document_classes(class_transaction_results_df):
    return class_transaction_results_df.loc[
        class_transaction_results_df["answer"].astype(str).str.strip().str.lower() == "yes",
        ["document_class", "node_id"],
    ].drop_duplicates().reset_index(drop=True)

def process_class_batch(batch_df, search_document_name, search_context=None):
    payload = generate_class_transaction_payload(batch_df, search_document_name, search_context)
    results = send_transaction_selection(payload)
    if not isinstance(results, list):
        raise Exception("Expected document-class transaction-selection output to be a list.")
    if len(results) != len(batch_df):
        raise Exception(
            "Document-class transaction-selection result count mismatch. "
            f"Input: {len(batch_df)}, Output: {len(results)}"
        )
    response_df = pd.DataFrame(results)
    return pd.concat(
        [batch_df.reset_index(drop=True), response_df.reset_index(drop=True)],
        axis=1,
    )

def process_class_batch_worker(batch_number, batch_df, search_document_name, search_context=None):
    try:
        with print_lock:
            print(f"[CLASS WORKER] Batch {batch_number}: {len(batch_df)} document classes")
        result_df = process_class_batch(batch_df, search_document_name, search_context)
        return {"batch_number": batch_number, "data": result_df, "error": None}
    except Exception as e:
        with print_lock:
            print(f"[CLASS WORKER ERROR] Batch {batch_number}: {str(e)}")
        return {"batch_number": batch_number, "data": None, "error": str(e)}

def run_class_transaction_selection(classes_df, search_document_name, search_context=None):
    if classes_df.empty:
        return pd.DataFrame()
    batches = []
    total = len(classes_df)
    for start in range(0, total, CLASS_TRANSACTION_BATCH_SIZE):
        end = min(start + CLASS_TRANSACTION_BATCH_SIZE, total)
        batches.append((
            len(batches) + 1,
            classes_df.iloc[start:end].copy(),
        ))
    results = []
    successful = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=CLASS_TRANSACTION_MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_class_batch_worker,
                batch_number,
                batch_df,
                search_document_name,
                search_context,
            ): batch_number
            for batch_number, batch_df in batches
        }
        for future in as_completed(futures):
            result = future.result()
            if result["error"]:
                failed += 1
            else:
                successful += 1
                results.append(result["data"])
    print(f"Document-class transaction batches: {len(batches)}")
    print(f"Successful: {successful}, Failed: {failed}")
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)

# ============================================================
# EXISTING RELATED DOCUMENT CLASS LOOKUP
# ============================================================
def get_related_document_classes(related_documents_df, search_context=None):
    if related_documents_df.empty:
        return pd.DataFrame()
    document_names = (
        related_documents_df["document_name"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    document_names = document_names[document_names != ""].drop_duplicates().tolist()
    if not document_names:
        return pd.DataFrame()
    client = get_bq_client()
    query = f"""
        SELECT DISTINCT
            c.cluster_id AS document_class_cui,
            c.cluster_label AS document_class,
            m.MEMBER_NAME AS possible_document_name,
            m.MEMBER_ID
        FROM `{MEMBER_TABLE}` m
        JOIN `{MAP_TABLE}` b
            ON m.MEMBER_ID = b.MEMBER_ID
        JOIN `{CLUSTER_TABLE}` c
            ON b.cluster_id = c.cluster_id
        WHERE m.MEMBER_NAME IN UNNEST(@document_names)
        ORDER BY c.cluster_id, m.MEMBER_NAME
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("document_names", "STRING", document_names)
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    if not df.empty:
        search_document_name = related_documents_df["search_document_name"].iloc[0]
        df.insert(0, "search_document_name", search_document_name)
        if search_context:
            df.insert(1, "search_context", search_context)
    return df

# ============================================================
# PIPELINE RUNNER FOR EACH (TEXT, CONTEXT) PAIR
# ============================================================
def process_search_pair(
    search_document_name,
    search_context,
    summary_accumulator,
    selected_class_dataframe_accumulator,
):
    """Executes the pipeline for a single pair.

    Extracts selected document classes for the summary JSON and a DataFrame.
    """
    print("\n========================================")
    print("SEARCH")
    print("========================================")
    print(f"Search document: {search_document_name}")
    print(f"Search context : {search_context if search_context else '[None provided]'}")

    selected_activities = set()
    selected_document_names = set()
    selected_class_details = []

    # STEP 1 - CLUSTER SELECTION
    print("\n========================================")
    print("STEP 1 - CLUSTER SELECTION")
    print("========================================")
    probable_documents = get_probable_documents(search_document_name, search_context)
    if not probable_documents:
        print("No probable documents were found.")
        summary_accumulator.append({
            "text": search_document_name,
            "context": search_context,
        })
        return

    # STEP 2 - GET ACTIVITIES
    print("\n========================================")
    print("STEP 2 - GET ACTIVITIES")
    print("========================================")
    activities_df = get_activities_for_documents(probable_documents)
    if activities_df.empty:
        print("No valid activities found.")
        summary_accumulator.append({
            "text": search_document_name,
            "context": search_context,
        })
        return

    print(f"Candidate activities: {len(activities_df)}")

    # STEP 3 - ACTIVITY TRANSACTION SELECTION (STAGE 1)
    print("\n========================================")
    print("STEP 3 - ACTIVITY TRANSACTION SELECTION")
    print("========================================")
    activity_results_df = run_activity_transaction_selection(
        activities_df,
        search_document_name,
        search_context,
    )
    if activity_results_df.empty:
        print("No activity transaction results were produced.")
        summary_accumulator.append({
            "text": search_document_name,
            "context": search_context,
        })
        return

    print("Activity transaction results:")
    print(activity_results_df["answer"].value_counts(dropna=False).to_string())

    yes_act_df = activity_results_df[
        activity_results_df["answer"].astype(str).str.strip().str.lower() == "yes"
    ]
    if not yes_act_df.empty and "activity_name" in yes_act_df.columns:
        selected_activities = set(
            yes_act_df["activity_name"].dropna().astype(str).str.strip().tolist()
        )
        selected_activities.discard("")

    # STEP 4 - GET CANDIDATE RELATED DOCUMENT NAMES
    print("\n========================================")
    print("STEP 4 - CANDIDATE RELATED DOCUMENT NAMES")
    print("========================================")
    related_documents_df = get_related_document_names(
        activity_results_df,
        search_document_name,
        search_context,
    )
    if related_documents_df.empty:
        print("No candidate related document names found.")
        summary_accumulator.append({
            "text": search_document_name,
            "context": search_context,
        })
        return

    print(f"Candidate related documents: {len(related_documents_df)}")

    # STEP 5 - DOCUMENT NAME TRANSACTION SELECTION (STAGE 2)
    print("\n========================================")
    print("STEP 5 - DOCUMENT NAME TRANSACTION SELECTION")
    print("========================================")
    document_transaction_results_df = run_document_transaction_selection(
        related_documents_df,
        search_document_name,
        search_context,
    )
    if document_transaction_results_df.empty:
        print("No document-name transaction results were produced.")
        summary_accumulator.append({
            "text": search_document_name,
            "context": search_context,
        })
        return

    print("Document-name transaction results:")
    print(
        document_transaction_results_df["answer"]
        .value_counts(dropna=False)
        .to_string()
    )

    yes_documents_df = document_transaction_results_df[
        document_transaction_results_df["answer"].astype(str).str.strip().str.lower() == "yes"
    ].copy()
    if yes_documents_df.empty:
        print("No document names were classified as related.")
        summary_accumulator.append({
            "text": search_document_name,
            "context": search_context,
        })
        return

    yes_document_names = (
        yes_documents_df["document_name"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    yes_document_names = yes_document_names[yes_document_names != ""].drop_duplicates().tolist()
    selected_document_names = set(yes_document_names)
    print(f"Documents classified Yes for class lookup: {len(yes_document_names)}")

    # STEP 6 - FETCH DOCUMENT CLASSES
    print("\n========================================")
    print("STEP 6 - DOCUMENT-CLASS DISCOVERY")
    print("========================================")
    classes_df = get_document_classes(yes_document_names)
    if classes_df.empty:
        print("No document classes found.")
        summary_accumulator.append({
            "text": search_document_name,
            "context": search_context,
        })
        return

    classes_df.insert(0, "search_document_name", search_document_name)
    if search_context:
        classes_df.insert(1, "search_context", search_context)

    print(f"Document classes discovered: {len(classes_df)}")

    # STEP 7 - DOCUMENT-CLASS TRANSACTION SELECTION (STAGE 3)
    print("\n========================================")
    print("STEP 7 - DOCUMENT-CLASS TRANSACTION SELECTION")
    print("========================================")
    class_transaction_results_df = run_class_transaction_selection(
        classes_df,
        search_document_name,
        search_context,
    )
    if class_transaction_results_df.empty:
        print("No document-class transaction results were produced.")
        summary_accumulator.append({
            "text": search_document_name,
            "context": search_context,
        })
        return

    print("Document-class transaction results:")
    print(
        class_transaction_results_df["answer"]
        .value_counts(dropna=False)
        .to_string()
    )

    selected_document_classes_df = select_document_classes(
        class_transaction_results_df
    )
    if not selected_document_classes_df.empty:
        selected_class_dataframe_accumulator.append(selected_document_classes_df)
        selected_class_details = selected_document_classes_df[
            ["document_class", "node_id"]
        ].to_dict(orient="records")

    # Record summary for this pair
    summary_accumulator.append({
        "text": search_document_name,
        "context": search_context,
        "selected_document_class_details": selected_class_details,
    })

    print("\n========================================")
    print("PAIR IN-MEMORY PROCESSING COMPLETE")
    print("========================================")
    print(f"Selected Activities: {len(selected_activities)}")
    print(f"Selected Documents : {len(selected_document_names)}")
    print(f"Selected Classes   : {len(selected_class_details)}")


# ============================================================
# MAIN
# ============================================================
def main():
    gcp_update_header()

    if not os.path.exists(TEST_PAIRS_FILE):
        raise FileNotFoundError(
            f"Test pairs file '{TEST_PAIRS_FILE}' not found. "
            "Please ensure you generate it first."
        )

    with open(TEST_PAIRS_FILE, "r", encoding="utf-8") as f:
        test_pairs = json.load(f)

    if not isinstance(test_pairs, list) or len(test_pairs) == 0:
        print("No test pairs found in the file.")
        return

    print(f"Loaded {len(test_pairs)} test pair(s) from {TEST_PAIRS_FILE}.\n")

    # Summary accumulator for JSON file output
    summary_accumulator = []
    selected_class_dataframe_accumulator = []

    for index, item in enumerate(test_pairs, start=1):
        search_document_name = item.get("text", "").strip()
        search_context = item.get("context", "").strip() or None

        if not search_document_name:
            print(f"[SKIP] Pair #{index} has an empty 'text'. Skipping.")
            continue

        print(f"\n############################################################")
        print(f"PROCESSING PAIR {index}/{len(test_pairs)}")
        print(f"############################################################")

        try:
            process_search_pair(
                search_document_name,
                search_context,
                summary_accumulator,
                selected_class_dataframe_accumulator,
            )
        except Exception as e:
            with print_lock:
                print(f"[ERROR] Failed processing pair #{index} ('{search_document_name}'): {str(e)}")
            summary_accumulator.append({
                "text": search_document_name,
                "context": search_context,
                "error": str(e)
            })

    selected_document_classes_df = (
        pd.concat(selected_class_dataframe_accumulator, ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
        if selected_class_dataframe_accumulator
        else pd.DataFrame(columns=["document_class", "node_id"])
    )
    print(
        "Selected document-class rows retained in DataFrame: "
        f"{len(selected_document_classes_df)}"
    )

    # ========================================================
    # SAVE SUMMARY JSON
    # ========================================================
    print("\n========================================")
    print("SAVING SELECTED ENTITIES SUMMARY JSON")
    print("========================================")
    with open(SUMMARY_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_accumulator, f, indent=2)
    print(f"Summary successfully written to: {SUMMARY_OUTPUT_FILE}")

    print("\n========================================")
    print("ALL TEST PAIRS PROCESSED SUCCESSFULLY")
    print("========================================")
    return selected_document_classes_df


if __name__ == "__main__":
    main()
