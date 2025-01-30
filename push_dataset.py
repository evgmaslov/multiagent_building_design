from datasets import load_dataset, Dataset
import json
from tqdm.auto import tqdm
from huggingface_hub import login
import argparse

def main(json_path, dataset_name, subset_name):
    login()
    
    with open(json_path, "r") as f:
        raw_dataset = json.load(f)
    
    dataset = {}
    for key in raw_dataset[0].keys():
        dataset[key] = []
    
    for sample in raw_dataset:
        for key in sample.keys():
            if type(sample[key]) == type(str):
                value = sample[key]
            else:
                value = json.dumps(sample[key])
            dataset[key].append(value)
    
    dataset = Dataset.from_dict(dataset)
    dataset.push_to_hub(dataset_name, subset_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--json_path", default="dataset.json", type=str)
    parser.add_argument(f"--dataset_name", default="evgmaslov/flats", type=str)
    parser.add_argument(f"--subset_name", default="new_subset", type=str)
    args = parser.parse_args()
    
    main(args.json_path, args.dataset_name, args.subset_name)
    