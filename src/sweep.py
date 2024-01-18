import wandb
import yaml

from src.train import train

# sweep configuration
file = open("src/conf/sweep_config.yaml", "r")
sweep_configuration = yaml.load(file, Loader=yaml.FullLoader)

sweep_id = wandb.sweep(sweep_configuration, entity="mlops_team29", project="mlops-project")

# run the sweep
wandb.agent(sweep_id, function=train)
