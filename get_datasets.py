
import pandas as pd
from datasets import load_dataset

def get_prompt_attack_dataset(n: int) -> pd.DataFrame:
    samples_per_category = n // 4
    rs = 42
    
    # SPML dataset
    dataset_1 = load_dataset("reshabhs/SPML_Chatbot_Prompt_Injection", trust_remote_code=True).get("train").to_pandas()
    spml_samples = dataset_1.sample(samples_per_category, random_state=rs)
    spml_samples = spml_samples.rename(columns={"Prompt injection": "label", "User Prompt": "text"})
    spml_samples["category"] = "SPML"

    # Hackaprompt dataset (all positive examples)
    dataset_2 = load_dataset("hackaprompt/hackaprompt-dataset", trust_remote_code=True).get("train").to_pandas()
    hackaprompt_samples = dataset_2.query("correct == True").sample(samples_per_category, random_state=rs)
    hackaprompt_samples = hackaprompt_samples.rename(columns={"user_input": "text"})
    hackaprompt_samples["category"] = "hackaprompt"
    hackaprompt_samples["label"] = 1 # Since we filter by correct == True, these are all prompt inject attacks that have succeeded on GPT-3.5 and FlanT5

    # Bitext dataset (all negative examples)
    dataset_3 = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", trust_remote_code=True).get("train").to_pandas()
    bitext_samples = dataset_3.query("intent == 'get_refund'").sample(samples_per_category, random_state=rs)
    bitext_samples = bitext_samples.rename(columns={"instruction": "text"})
    bitext_samples["category"] = "bitext"
    bitext_samples["label"] = 0 # Assuming that this doesn't contain any prompt injection requests

    # Wildjailbreak dataset
    # dataset_4 = load_dataset("allenai/wildjailbreak", "eval", trust_remote_code=True).get("train").to_pandas()
    # wildjailbreak_samples = dataset_4.sample(samples_per_category, random_state=rs)
    # wildjailbreak_samples = wildjailbreak_samples[["label", "adversarial"]]
    # wildjailbreak_samples = wildjailbreak_samples.rename(columns={"adversarial": "text"})
    # wildjailbreak_samples["category"] = "wildjailbreak"
    # wildjailbreak_samples["label"] = wildjailbreak_samples["label"].astype(int)

    return pd.concat([
        spml_samples,
        hackaprompt_samples,
        bitext_samples, 
        # wildjailbreak_samples
    ])[["category", "text", "label"]]

def print_dataset_summary(df):
    total = len(df)
    print(f"Total samples: {total}")
    print("\nLabel distribution:")
    print(df.groupby('label').size().to_string())
    print("\nCategory distribution:")
    print(df.groupby(['category', 'label']).size().to_string())

if __name__ == "__main__":
    sample = get_prompt_attack_dataset(200)
    print_dataset_summary(sample)
    sample.to_csv("prompt_attack_dataset.csv", index=False)