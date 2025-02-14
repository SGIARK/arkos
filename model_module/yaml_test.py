import yaml

path = "../config_module/config.yaml"
with open(path, "r") as file:
    info = yaml.safe_load(file)

print(info["model_path"])
