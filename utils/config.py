import argparse

import yaml


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to the config file")
    return parser.parse_args()
