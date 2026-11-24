
import os
import sys

from dotenv import load_dotenv

from networksecurity.exceptions.exception import NetworkSecurityException
from networksecurity.loggings.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models
)
from networksecurity.utils.ml_utils.metric.classification_metric import (
    get_classification_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

# =========================
# MLOps Imports (Disabled)
# =========================

# import mlflow
# import mlflow.sklearn
# from urllib.parse import urlparse
# import dagshub


load_dotenv()

# =========================
# DagsHub / MLflow Configuration (Disabled)
# =========================

# os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
# os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
# os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")


class ModelTrainer:

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # =========================
    # MLflow Tracking (Disabled)
    # =========================

    # def track_mlflow(self, best_model, classificationmetric, run_name):

    #     try:

    #         mlflow.set_tracking_uri(
    #             os.environ["MLFLOW_TRACKING_URI"]
    #         )

    #         mlflow.set_registry_uri(
    #             os.environ["MLFLOW_TRACKING_URI"]
    #         )

    #         mlflow.set_experiment("Cyber Threat Detection")

    #         tracking_url_type_store = urlparse(
    #             mlflow.get_tracking_uri()
    #         ).scheme

    #         with mlflow.start_run(run_name=run_name):

    #             f1_score = classificationmetric.f1_score
    #             precision_score = classificationmetric.precision_score
    #             recall_score = classificationmetric.recall_score

    #             mlflow.log_metric("f1_score", f1_score)
    #             mlflow.log_metric("precision", precision_score)
    #             mlflow.log_metric("recall", recall_score)

    #             mlflow.log_param(
    #                 "model_name",
    #                 type(best_model).__name__
    #             )

    #             if tracking_url_type_store != "file":

    #                 mlflow.sklearn.log_model(
    #                     best_model,
    #                     artifact_path="model",
    #                     registered_model_name="cyber-threat-detection-model"
    #                 )

    #             else:

    #                 mlflow.sklearn.log_model(
    #                     best_model,
    #                     artifact_path="model"
    #                 )

    #             logging.info(
    #                 f"MLflow tracking completed for {run_name}"
    #             )

    #     except Exception as e:
    #     logging.error(f"MLflow tracking failed: {str(e)}")
    #     raise NetworkSecurityException(e, sys)

    def train_model(self, X_train, y_train, x_test, y_test):

        try:

            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {

                "Decision Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                },

                "Random Forest": {
                    'n_estimators': [8, 16, 32, 128, 256]
                },

                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                "Logistic Regression": {},

                "AdaBoost": {
                    'learning_rate': [.1, .01, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }

            }

            # Evaluate all models
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=x_test,
                y_test=y_test,
                models=models,
                param=params
            )

            # To get best model score from dict
            best_model_score = max(model_report.values())

            # To get best model name from dict
            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model = models[best_model_name]

            logging.info(
                f"Best model found: {best_model_name}"
            )

            logging.info(
                f"Best model score: {best_model_score}"
            )

            # Ensure the selected model is fitted
            best_model.fit(X_train, y_train)

            # Prediction on training data
            y_train_pred = best_model.predict(X_train)

            classification_train_metric = get_classification_score(
                y_true=y_train,
                y_pred=y_train_pred
            )

            # =========================
            # MLflow Training Tracking (Disabled)
            # =========================

            # self.track_mlflow(
            #     best_model,
            #     classification_train_metric,
            #     "Training Metrics"
            # )

            # Prediction on testing data
            y_test_pred = best_model.predict(x_test)

            classification_test_metric = get_classification_score(
                y_true=y_test,
                y_pred=y_test
            )

            # =========================
            # MLflow Testing Tracking (Disabled)
            # =========================

            # self.track_mlflow(
            #     best_model,
            #     classification_test_metric,
            #     "Testing Metrics"
            # )

            # Load preprocessor object
            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            # Create model directory
            model_file_path = self.model_trainer_config.trained_model_file_path

            model_dir_path = os.path.dirname(
                os.path.abspath(model_file_path)
            )

            os.makedirs(
                model_dir_path,
                exist_ok=True
            )

            # Combine preprocessor and trained model
            network_model = NetworkModel(
                preprocessor=preprocessor,
                model=best_model
            )

            # Save trained model with preprocessor
            save_object(
                file_path=model_file_path,
                obj=network_model
            )

            # Model pusher
            final_model_path = os.path.join(
                "final_model",
                "model.pkl"
            )

            os.makedirs(
                os.path.dirname(final_model_path),
                exist_ok=True
            )

            save_object(
                file_path=final_model_path,
                obj=network_model
            )

            # Model Trainer Artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )

            logging.info(
                f"Model trainer artifact: {model_trainer_artifact}"
            )

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        try:

            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )

            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )

            # Loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(
                x_train,
                y_train,
                x_test,
                y_test
            )

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
