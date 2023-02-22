import torch
from torch import nn
import numpy as np
from pathlib import Path
import typing
import copy
import shutil
from recap import CfgNode as CN

from chess_recognition.utils import device, build_dataset, build_data_loader, Datasets
from chess_recognition.occupancy_classifier_models import get_occupancy_classifier
from chess_recognition.piece_classifier_models import get_piece_classifier


# Model that is to be trained
MODEL_NAME = "CNN100_3Conv_3Pool_3FC" 
# (needs an associated .yaml file in config/ 
# and a corresponding classifier model from 
# occupancy_classifier_models.py or piece_classifier_models.py)


CONFIGS_DIR = Path("./config")
OUT_DIR = Path("../chess_recognition/data/occupancy")

def train(name: str) -> nn.Module:
    """Prepares for training and loads config + model

    Args:
        name (str): occupancy_classifier or piece_classifier

    Returns:
        nn.Module: the trained model
    """
    print(f"Training {MODEL_NAME}")

    configs_dir = Path(f'{CONFIGS_DIR}/{name}')
    run_dir = Path(f"runs/{name}/{MODEL_NAME}")
    cfg = CN.load_yaml_with_base(f"{configs_dir}/{MODEL_NAME}.yaml")
    if (name == "occupancy_classifier"):
        model = get_occupancy_classifier(MODEL_NAME)
    elif (name == "piece_classifier"):
        model = get_piece_classifier(MODEL_NAME)
    else:
        print(f"Couldn't open the {name} {MODEL_NAME}")

    train_model(cfg, run_dir, model)


def train_model(cfg: CN, run_dir: Path, model: torch.nn.Module) -> nn.Module:
    """Actual training of the model

    Args:
        cfg (CN): config of the model
        run_dir (Path): Path where to save the trained model
        model (torch.nn.Module): model that is to be trained

    Returns:
        nn.Module: _description_
    """
    print(f"Starting training in {run_dir}")
    model_name = run_dir.name

    # Create folder
    if run_dir.exists():
        print(f"WARNING - The folder {run_dir} already exists and will be overwritten by this run")
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Store config
    with (Path(f"{run_dir}/{model_name}.yaml")).open("w") as f:
        cfg.dump(stream=f)

    # Move model to device
    device(model)

    best_weights, best_accuracy, best_step = copy.deepcopy(
        model.state_dict()), 0., 0

    criterion = nn.CrossEntropyLoss()

    modes = {Datasets.TRAIN, Datasets.VAL}

    datasets = {mode: build_dataset(cfg, mode)
                for mode in modes}
    classes = [datasets[Datasets.TRAIN].classes]
    loader = {mode: build_data_loader(cfg, datasets[mode], mode)
              for mode in modes}


    def perform_iteration(data: typing.Tuple[torch.Tensor, torch.Tensor], mode: Datasets):
        inputs, labels = map(device, data)
        with torch.set_grad_enabled(mode == Datasets.TRAIN):
            # Reset gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if mode == Datasets.TRAIN:
                loss.backward()

        # Perform optimization
        if mode == Datasets.TRAIN:
            optimizer.step()

        # Return
        return loss.item()

    step = 0

    # Ensure we're in training mode
    model.train()

    # Loop over training phases
    for phase in cfg.TRAINING.PHASES:

        for p in model.parameters():
            p.requires_grad = False
        parameters = list(model.parameters()) if phase.PARAMS == "all" \
            else model.params[phase.PARAMS]
        for p in parameters:
            p.requires_grad = True
        
        optimizer = torch.optim.Adam(parameters, lr=0.0001)

        # Loop over epochs (passes over the whole dataset)
        for epoch in range(phase.EPOCHS):
            # Iterate the training set
            losses = []
            for i, data in enumerate(loader[Datasets.TRAIN]):

                # Perform training iteration
                losses.append(perform_iteration(data, mode=Datasets.TRAIN))

                # Logging every 100 stepts
                if step % 100 == 0:
                    loss = np.mean(list(losses))        
                    print(f"Step {step:5d}: loss {loss:.3f}")

                # Save weights if we get a better performance
                confusion_matrix = np.zeros((len(classes), len(classes)),dtype=np.uint32)
                correct = np.trace(confusion_matrix)
                total = np.sum(confusion_matrix)
                
                accuracy = correct/total if total != 0 else 0
                if accuracy >= best_accuracy:
                    best_accuracy = accuracy
                    best_weights = copy.deepcopy(model.state_dict())
                    best_step = step

                # Get ready for next step
                step += 1

    print("Finished training")
    print(f"Restoring best weight state (step {best_step} with validation accuracy of {best_accuracy})")
    model.load_state_dict(best_weights)
    torch.save(model, f"{run_dir}/{model_name}.pt")
    with (Path(f"{run_dir}/{model_name}.txt")).open("w") as f:
        f.write(f"exported at step: {best_step}\n")
    return model

