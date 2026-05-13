import json
import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import requests
from tqdm import tqdm

def load_category_candidates_from_json(json_path: str) -> List[Dict[str, Any]]:
    """
    从 JSON 中读取 clusters，构造类别候选。

    预期 JSON 结构:
    {
      "clusters": [
        {
          "cluster_id": 1,
          "category": "instrument",
          "terms": ["guitar", "string", "pedal"]
        },
        {
          "cluster_id": 2,
          "category": "sound",
          "terms": ["sound", "tone", "cable"]
        }
      ]
    }

    返回:
    [
      {
        "category": "instrument",
        "example_terms": ["guitar", "string", "pedal"]
      },
      {
        "category": "sound",
        "example_terms": ["sound", "tone", "cable"]
      }
    ]
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clusters = data.get("clusters", [])
    if not clusters:
        raise ValueError("No 'clusters' found in JSON file.")

    candidates: List[Dict[str, Any]] = []
    seen_categories = set()

    for cluster in clusters:
        category = cluster.get("category")
        terms = cluster.get("terms", [])

        if not category:
            continue

        category = str(category).strip()
        if not category or category in seen_categories:
            continue

        cleaned_terms = []
        seen_terms = set()

        for term in terms:
            if term is None:
                continue
            term = str(term).strip()
            if not term or term in seen_terms:
                continue
            cleaned_terms.append(term)
            seen_terms.add(term)

        candidates.append({
            "category": category,
            "example_terms": cleaned_terms
        })
        seen_categories.add(category)

    if not candidates:
        raise ValueError("No valid category candidates found in JSON file.")

    return candidates
def load_prompt_template(prompt_txt_path: str) -> str:
    """
    读取 prompt 模板文件。
    """
    path = Path(prompt_txt_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {prompt_txt_path}")
    return path.read_text(encoding="utf-8")


def render_prompt(
    template: str,
    term: str,
    candidates: List[Dict[str, Any]],
) -> str:
    """
    将 prompt 模板中的占位符替换为当前术语和类别候选。

    支持以下占位符：
    - {term}
    - {candidates_json}
    """
    candidates_json = json.dumps(candidates, ensure_ascii=False, indent=2)

    rendered = template.replace("{term}", term)
    rendered = rendered.replace("{candidates_json}", candidates_json)
    return rendered


def extract_json_from_response(raw_text: str) -> Dict[str, Any]:
    """
    从 Ollama 返回文本中解析 JSON。
    """
    raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # 兼容 ```json ... ``` 包裹
    import re
    code_block_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL)
    if code_block_match:
        return json.loads(code_block_match.group(1))

    json_match = re.search(r"(\{.*\})", raw_text, flags=re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    raise ValueError(f"Failed to parse JSON from Ollama response:\n{raw_text}")


def call_ollama_with_prompt(
    prompt: str,
    model: str = "qwen2.5:7b",
    base_url: str = "http://localhost:11434",
    timeout: int = 300,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    调用 Ollama，返回解析后的 JSON。
    """
    url = f"{base_url}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature
        }
    }

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()

    raw_text = response.json().get("response", "").strip()
    return extract_json_from_response(raw_text)


def classify_terms_with_ollama(
    csv_path: str,
    json_path: str,
    prompt_txt_path: str,
    term_column: str,
    output_csv_path: str,
    at2ac_path:str,
    model: str = "qwen2.5:7b",
    base_url: str = "http://localhost:11434",
    timeout: int = 300,
    temperature: float = 0.0,
    sleep_seconds: float = 0.0,
    save_rendered_prompts: bool = False,
    rendered_prompt_dir: str = "rendered_prompts",
) -> pd.DataFrame:
    """
    对 CSV 中的每一个术语调用一次 Ollama。
    Prompt 从 txt 文件读取，并在每次调用时填入：
    - 当前术语
    - 候选类别 JSON
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_file)

    if term_column not in df.columns:
        raise ValueError(f"Column '{term_column}' not found in CSV.")

    candidates = load_category_candidates_from_json(json_path)
    prompt_template = load_prompt_template(prompt_txt_path)
    valid_categories = [item["category"] for item in candidates]

    if save_rendered_prompts:
        Path(rendered_prompt_dir).mkdir(parents=True, exist_ok=True)

    assigned_categories = []
    raw_results = []
    # used_prompts = []

    
    
    all_aspect_terms = df[term_column].dropna().unique().tolist()
    total = len(all_aspect_terms)
    
    # for idx, term in enumerate(df[term_column].unique().tolist(), start=1):
    for idx, term in enumerate(tqdm(all_aspect_terms, desc="Classifying terms",ncols=100), start=1):

        if pd.isna(term) or str(term).strip() == "":
            assigned_categories.append("")
            raw_results.append("")
            # used_prompts.append("")
            continue

        term_str = str(term).strip()
        prompt = render_prompt(
            template=prompt_template,
            term=term_str,
            candidates=candidates,
        )

        if save_rendered_prompts:
            safe_name = f"{idx:04d}.txt"
            Path(rendered_prompt_dir, safe_name).write_text(prompt, encoding="utf-8")

        result = call_ollama_with_prompt(
            prompt=prompt,
            model=model,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
        )

        category = result.get("category")
        if category not in valid_categories:
            # raise ValueError(
            #     f"Invalid category returned for term '{term_str}': {category}. "
            #     f"Valid categories: {valid_categories}"
            # )
            # print('New category')
            valid_categories.append(category)
        assigned_categories.append(category)
        raw_results.append(json.dumps(result, ensure_ascii=False))
        # used_prompts.append(prompt)

        # print(f"[{idx}/{total}] {term_str} -> {category}")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    at2ac=dict()
    for at,ac in zip(all_aspect_terms,assigned_categories):
        at2ac[at]=ac
    
    all_aspect_categories = [at2ac.get(str(term).strip(), "") for term in df[term_column]]
    df["assigned_category"] = all_aspect_categories
    # df["ollama_raw_result"] = raw_results

    # 如果不想把完整 prompt 写进 csv，可以删掉这一列
    # df["used_prompt"] = used_prompts

    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    df2=pd.DataFrame(list(at2ac.items()),columns=['AT','AC'])
    df2.to_csv(at2ac_path,index=False)
    return df


if __name__ == "__main__":
    JSON_PATH = "src_ACG2/AC_res/Musical_Instruments_5/llama3.1_8b_clusters.json"
    CSV_PATH = "tmp_data/ACOS_result/Musical_Instruments_5/Musical_Instruments_5.csv"
    PROMPT_TXT_PATH = "src_ACG2/LLM_extraction/category_dection_prompt2.txt"
    TERM_COLUMN = "AT"
    OUTPUT_CSV_PATH = "src_ACG2/LLM_extraction/res.csv"
    AT2AC_PATH = "src_ACG2/LLM_extraction/at2ac.csv"
    result_df = classify_terms_with_ollama(
        csv_path=CSV_PATH,
        json_path=JSON_PATH,
        prompt_txt_path=PROMPT_TXT_PATH,
        term_column=TERM_COLUMN,
        output_csv_path=OUTPUT_CSV_PATH,
        at2ac_path=AT2AC_PATH,
        model="llama3.1:8b",
        base_url="http://localhost:11434",
        timeout=300,
        temperature=0.0,
        sleep_seconds=0.0,
        save_rendered_prompts=False,
    )

    print("\nDone.")
    print(result_df.head())