from pathlib import Path

# TODO weights and biases

# Folders and file paths
data_folder = Path(__file__).parent.resolve() / "data/"
data_folder.mkdir(parents=True, exist_ok=True)

eval_path = data_folder / "evaluation.csv"
assert eval_path.exists(), "You need to download the evaluation set from Kaggle"
train_path = data_folder / "train.csv"
assert train_path.exists(), "You need to download the train set from Kaggle"
glove_path = data_folder / "GoogleNews-vectors-negative300.bin.gz"


