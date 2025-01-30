import argparse
import json

from src.modules import MultiagentSystem

def main(config_path, output_path, log_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    multiagent_system = MultiagentSystem(config, log_path)
    user_input = input("Input: ")
    final_context = multiagent_system(user_input)
    with open(output_path, "w+") as f:
        json.dump(final_context, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--config_path", default="config.json", type=str)
    parser.add_argument(f"--output_path", default="output.json", type=str)
    parser.add_argument(f"--log_path", default="log.json", type=str)
    args = parser.parse_args()
    main(args.config_path, args.output_path, args.log_path)