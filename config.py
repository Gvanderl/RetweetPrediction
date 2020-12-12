from pathlib import Path

# TODO weights and biases

# Folders and file paths
project_path = Path(__file__).parent.resolve()
data_folder = project_path / "data/"
data_folder.mkdir(parents=True, exist_ok=True)
output_folder = project_path / "outputs/"
data_folder.mkdir(parents=True, exist_ok=True)

eval_path = data_folder / "evaluation.csv"
train_path = data_folder / "train.csv"
glove_path = data_folder / "GoogleNews-vectors-negative300.bin.gz"

model_folder = project_path / "saved_models"
model_folder.mkdir(parents=True, exist_ok=True)
