from aspect_category_extraction import build_and_run_pipeline_with_ollama
from aspect_category_dection import classify_terms_with_ollama

import pandas as pd
import os
def run(dataset,src_path,tmp_path,dst_path, at2ac_path,model="llama3.1:8b"):
    
    # path='tmp_data/ACOS_result/Musical_Instruments_5/Musical_Instruments_5.csv'
    
    df=pd.read_csv(src_path)
    
    result = build_and_run_pipeline_with_ollama(
        df=df,
        column="AT",
        prompt_txt_path="src_ACG2/LLM_extraction/prompt2.txt",
        top_k=100,
        model=model,
        base_url="http://localhost:11434",
        # prompt_output_path="tmp_data/LLM_AC_extraction/filled_prompt.txt",
        # raw_output_path="tmp_data/LLM_AC_extraction/{}_raw_output.txt".format(model.replace(":", "_")),
        # row_output_path=None,
        json_output_path=tmp_path,
        temperature=0.2,
        format_json=False,
    )

    # print(result["parsed_output"])

    result_df = classify_terms_with_ollama(
        csv_path=src_path,
        json_path=tmp_path,
        prompt_txt_path='src_ACG2/LLM_extraction/category_dection_prompt2.txt',
        term_column='AT',
        output_csv_path=dst_path,
        at2ac_path=at2ac_path,
        model=model,
        base_url="http://localhost:11434",
        timeout=600,
        temperature=0.0,
        sleep_seconds=0.2,
        save_rendered_prompts=False,
    )


if __name__ == "__main__":

    datasets = [
            # "Musical_Instruments_5",
            # "Office_Products_5",
            # "Toys_and_Games_5",
            # "Video_Games_5",
            "Beauty_5",
            # "yelp2",
        ]
    # 'vicuna:13b','qwen3.5:9b','qwen3:8b','qwen2.5:7b',
    for dataset in datasets:
        for model in ["llama3.1:8b",'glm4:latest',]:
            src_path="tmp_data/ACOS_result/{0}/{0}2.csv".format(dataset)
            tmp_path="src_ACG2/AC_res/{}/{}_clusters.json".format(dataset,model.replace(":", "_"))
            dst_path="src_ACG2/AC_res/{}/{}_classified.csv".format(dataset,model.replace(":", "_"))
            at2ac_path="src_ACG2/AC_res/{}/{}_at2ac.csv".format(dataset,model.replace(":", "_"))
            if os.path.exists(at2ac_path):
                print("File {} already exists. Skipping...".format(dst_path))
                continue
            print("Running dataset: {}, model: {}".format(dataset, model))

            dir = os.path.dirname(dst_path)
            if not os.path.exists(dir):
                os.makedirs(dir)
            run(dataset,src_path,tmp_path,dst_path,at2ac_path,model=model)