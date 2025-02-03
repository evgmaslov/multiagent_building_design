import argparse
import json
from datasets import load_dataset

from src.modules import MultiagentSystem

def main(config_path, dataset_name, subset_name, input_key, output_path, log_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    multiagent_system = MultiagentSystem(config, log_path)
    
    dataset = load_dataset(dataset_name, subset_name)["train"]
    submit = []
    with open(output_path, "r") as f:
        submit = json.load(f)
    for i, row in enumerate(dataset):
        user_input = row[input_key]
        final_context = multiagent_system(user_input)
        submit.append(final_context)
        with open(output_path, "w+") as f:
            json.dump(submit, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--config_path", default="config.json", type=str)
    parser.add_argument(f"--dataset_name", default="evgmaslov/flats", type=str)
    parser.add_argument(f"--subset_name", default="handwritten", type=str)
    parser.add_argument(f"--input_key", default="task", type=str)
    parser.add_argument(f"--output_path", default="output.json", type=str)
    parser.add_argument(f"--log_path", default="log.json", type=str)
    args = parser.parse_args()
    main(args.config_path, args.dataset_name, args.subset_name, args.input_key, args.output_path, args.log_path)