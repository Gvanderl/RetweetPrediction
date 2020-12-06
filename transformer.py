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
dp.load_csv(train_path, 10000)
dp.clean_df()
dp.norm_label()

X_train, X_test, y_train, y_test = dp.get_split_df()

train_df = pd.DataFrame({"text": X_train, "labels": y_train})
eval_df = pd.DataFrame({"text": X_test, "labels": y_test})

# Enabling regression
# Setting optional model configuration
model_args = ClassificationArgs()
model_args.num_train_epochs = 1
model_args.regression = True

# Create a ClassificationModel
model = ClassificationModel(
    "roberta",
    "roberta-base",
    num_labels=1,
    args=model_args
)

# Train the model
model.train_model(train_df, args={"overwrite_output_dir": True})

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

predictions, raw_outputs = model.predict(eval_df["text"])
predictions = dp.unnorm(predictions)
labels = dp.unnorm(eval_df["labels"])
mae = mean_absolute_error(y_true=labels, y_pred=predictions)
print("Prediction error:", mae)
