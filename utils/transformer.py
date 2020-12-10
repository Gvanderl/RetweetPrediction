from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
from dataprocessor import DataProcessor
from config import *
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

dp = DataProcessor()
dp.load_csv(train_path, 100000)
dp.clean_df()
# dp.norm_label()

X_train, X_test, y_train, y_test = dp.get_split_df()

train_df = pd.DataFrame({"text": X_train, "labels": y_train}).reset_index(drop=True)
eval_df = pd.DataFrame({"text": X_test, "labels": y_test}).reset_index(drop=True)

# Enabling regression
# Setting optional model configuration
model_args = ClassificationArgs()
model_args.train_batch_size = 48
model_args.eval_batch_size = 48
model_args.num_train_epochs = 1
model_args.regression = True
model_args.use_early_stopping = True
model_args.overwrite_output_dir = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 200
model_args.wandb_project = "RetweetPrediction"

# Create a ClassificationModel
model = ClassificationModel(
    "roberta",
    "roberta-base",
    num_labels=1,
    args=model_args,
)

# Train the model
step, training_progress = model.train_model(train_df,
                                            eval_df=eval_df,
                                            args={"overwrite_output_dir": True, "evaluate_during_training": True})
print(f"Training progress : {training_progress}")

# Evaluate the model
result, predictions, wrong_predictions = model.eval_model(eval_df)
print(f"Eval results : {result}")
# print(f"Predictions before unnorm : {predictions[:10]}")
# predictions = dp.unnorm(predictions)
# print(f"Predictions after unnorm : {predictions[:10]}")
# labels = dp.unnorm(eval_df["labels"])
labels = eval_df["labels"]
for i in range(20):
    print(f"Predicted: {predictions[i]} \t True: {labels[i]}")
mae = mean_absolute_error(y_true=labels, y_pred=predictions)
print("Prediction error:", mae)

# Get results
# kaggle_dp = DataProcessor()
# kaggle_dp.load_csv(eval_path, 100)
# kaggle_dp.clean_df()
# predictions, raw_outputs = model.predict(kaggle_dp.df["text"])
