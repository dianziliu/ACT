import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


def get_top_k_terms(
    df: pd.DataFrame,
    column: str,
    k: int = 100,
    dropna: bool = True,
    normalize_text: bool = True,
    lowercase: bool = False,
) -> List[str]:
    """
    从 DataFrame 的某一列中提取出现频率最高的前 k 个值。
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    series = df[column]

    if dropna:
        series = series.dropna()

    series = series.astype(str)

    if normalize_text:
        series = series.str.strip()
        series = series[series != ""]

    if lowercase:
        series = series.str.lower()

    top_terms = series.value_counts().head(k).index.tolist()
    return top_terms


def load_prompt_template(prompt_txt_path: str) -> str:
    """
    读取 prompt 模板文件。
    """
    path = Path(prompt_txt_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_txt_path}")
    return path.read_text(encoding="utf-8")


def render_prompt(
    template: str,
    terms: List[str],
    placeholder: str = "[Paste the list of 100 aspect terms here]",
    as_json_array: bool = True,
) -> str:
    """
    将 prompt 模板中的占位符替换为术语列表。
    支持两种占位符：
    1. [在此粘贴100个方面术语的列表]
    2. {terms_json}
    """
    replacement = json.dumps(terms, ensure_ascii=False, indent=2) if as_json_array else ", ".join(terms)

    if placeholder in template:
        return template.replace(placeholder, replacement)

    if "{terms_json}" in template:
        return template.replace("{terms_json}", replacement)

    raise ValueError(
        f"No supported placeholder found in prompt template. "
        f"Expected '{placeholder}' or '{{terms_json}}'."
    )


def call_ollama(
    prompt: str,
    model: str = "qwen2.5:7b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.2,
    timeout: int = 300,
    system_prompt: Optional[str] = None,
    format_json: bool = False,
) -> str:
    """
    调用本地 Ollama 模型。

    Args:
        prompt: 最终 prompt
        model: ollama 模型名，例如 qwen2.5:7b / llama3.1:8b / mistral:7b
        base_url: Ollama 服务地址
        temperature: 生成温度
        timeout: 请求超时秒数
        system_prompt: 可选系统提示词
        format_json: 是否请求 JSON 格式输出

    Returns:
        模型返回文本
    """
    url = f"{base_url}/api/generate"

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    if system_prompt:
        payload["system"] = system_prompt

    if format_json:
        payload["format"] = "json"

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    return data.get("response", "").strip()


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    从模型输出中提取 JSON。
    兼容：
    1. 纯 JSON
    2. ```json ... ``` 包裹
    3. 文本中夹带 JSON
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    code_block_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if code_block_match:
        return json.loads(code_block_match.group(1))

    json_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    raise ValueError("No valid JSON object found in model output.")


def save_json(data: Dict[str, Any], output_path: str) -> None:
    """
    保存 JSON 文件。
    """
    Path(output_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def build_and_run_pipeline_with_ollama(
    df: pd.DataFrame,
    column: str,
    prompt_txt_path: str,
    top_k: int = 100,
    model: str = "qwen2.5:7b",
    base_url: str = "http://localhost:11434",
    prompt_output_path: Optional[str] = None,
    raw_output_path: Optional[str] = None,
    json_output_path: Optional[str] = None,
    temperature: float = 0.2,
    format_json: bool = False,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    一键跑通整个 pipeline：
    - 提取 top_k 高频术语
    - 读取 prompt.txt
    - 替换 Input
    - 调用 Ollama 本地模型
    - 解析 JSON
    - 可选保存 prompt / raw output / json
    """
    terms = get_top_k_terms(df=df, column=column, k=top_k)
    template = load_prompt_template(prompt_txt_path)
    final_prompt = render_prompt(template=template, terms=terms)

    if prompt_output_path:
        Path(prompt_output_path).write_text(final_prompt, encoding="utf-8")

    raw_output = call_ollama(
        prompt=final_prompt,
        model=model,
        base_url=base_url,
        temperature=temperature,
        system_prompt=system_prompt,
        format_json=format_json,
    )

    if raw_output_path:
        Path(raw_output_path).write_text(raw_output, encoding="utf-8")

    parsed_output = extract_json_from_text(raw_output)

    if json_output_path:
        save_json(parsed_output, json_output_path)

    return {
        "terms": terms,
        "prompt": final_prompt,
        "raw_output": raw_output,
        "parsed_output": parsed_output,
    }
    
    
    
if __name__ == "__main__":


    path='tmp_data/ACOS_result/Musical_Instruments_5/Musical_Instruments_5.csv'
    df=pd.read_csv(path)
    model= "llama3.1:8b"
    result = build_and_run_pipeline_with_ollama(
        df=df,
        column="AT",
        prompt_txt_path="tmp_data/LLM_AC_extraction/prompt_ext.txt",
        top_k=100,
        model=model,
        base_url="http://localhost:11434",
        prompt_output_path="tmp_data/LLM_AC_extraction/filled_prompt.txt",
        raw_output_path="tmp_data/LLM_AC_extraction/{}_raw_output.txt".format(model.replace(":", "_")),
        json_output_path="tmp_data/LLM_AC_extraction/{}_clusters.json".format(model.replace(":", "_")),
        temperature=0.2,
        format_json=False,
    )

    print(result["parsed_output"])