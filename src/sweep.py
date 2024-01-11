import wandb

from src.train import train

sweep_configuration = {
    "name": "my-awesome-sweep",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "method": "grid",
    "parameters": {"lr": {"values": [0.01, 0.001]}},
}

sweep_id = wandb.sweep(sweep_configuration, entity="mlops_team29", project="MLOps_project")

# run the sweep
wandb.agent(sweep_id, function=train)