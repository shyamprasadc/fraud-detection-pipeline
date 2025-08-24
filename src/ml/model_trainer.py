"""
ML model trainer for fraud detection.
"""

import logging
import pickle
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
import xgboost as xgb

from ..models.transaction import Transaction
from ..processing.feature_engine import FeatureEngine


class FraudDetectionModelTrainer:
    """Trainer for fraud detection ML models."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the model trainer."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Model configuration
        self.model_type = self.config.get("model_type", "random_forest")
        self.model_path = self.config.get(
            "model_path", "models/fraud_detection_model.pkl"
        )
        self.scaler_path = self.config.get("scaler_path", "models/scaler.pkl")
        self.feature_names_path = self.config.get(
            "feature_names_path", "models/feature_names.json"
        )

        # Training configuration
        self.test_size = self.config.get("test_size", 0.2)
        self.random_state = self.config.get("random_state", 42)
        self.cv_folds = self.config.get("cv_folds", 5)

        # Model parameters
        self.model_params = self.config.get("model_params", {})

        # Initialize components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

        # Ensure model directory exists
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)

    def prepare_training_data(
        self, transactions: List[Transaction], feature_engine: FeatureEngine
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from transactions."""
        self.logger.info(
            f"Preparing training data from {len(transactions)} transactions"
        )

        features_list = []
        labels = []

        for transaction in transactions:
            try:
                # Calculate features
                features = feature_engine.calculate_features(transaction)

                # Get label
                label = 1 if transaction.is_fraud else 0

                features_list.append(list(features.values()))
                labels.append(label)

            except Exception as e:
                self.logger.warning(
                    f"Error processing transaction {transaction.transaction_id}: {e}"
                )
                continue

        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels)

        # Store feature names
        if features_list:
            self.feature_names = list(features_list[0].keys())

        self.logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        self.logger.info(f"Fraud rate: {np.mean(y):.3f}")

        return X, y

    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train the fraud detection model."""
        self.logger.info(f"Training {self.model_type} model")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create and train model
        self.model = self._create_model()

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=self.cv_folds
        )

        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        # Feature importance
        feature_importance = self._get_feature_importance()

        # Results
        results = {
            "model_type": self.model_type,
            "train_score": train_score,
            "test_score": test_score,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "metrics": metrics,
            "feature_importance": feature_importance,
            "training_date": datetime.utcnow().isoformat(),
            "n_samples": len(X),
            "n_features": X.shape[1],
            "fraud_rate": np.mean(y),
        }

        self.logger.info(f"Training completed. Test accuracy: {test_score:.3f}")
        self.logger.info(
            f"Cross-validation score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})"
        )

        return results

    def _create_model(self):
        """Create model based on configuration."""
        if self.model_type == "random_forest":
            default_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            params = {**default_params, **self.model_params}
            return RandomForestClassifier(**params)

        elif self.model_type == "xgboost":
            default_params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": self.random_state,
                "eval_metric": "logloss",
            }
            params = {**default_params, **self.model_params}
            return xgb.XGBClassifier(**params)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate model performance metrics."""
        metrics = {}

        # Basic metrics
        metrics["accuracy"] = np.mean(y_true == y_pred)
        metrics["precision"] = np.sum((y_true == 1) & (y_pred == 1)) / max(
            1, np.sum(y_pred == 1)
        )
        metrics["recall"] = np.sum((y_true == 1) & (y_pred == 1)) / max(
            1, np.sum(y_true == 1)
        )
        metrics["f1_score"] = (
            2
            * (metrics["precision"] * metrics["recall"])
            / max(1e-8, metrics["precision"] + metrics["recall"])
        )

        # AUC metrics
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        metrics["average_precision"] = average_precision_score(y_true, y_pred_proba)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)

        # Additional metrics
        metrics["false_positive_rate"] = fp / max(1, fp + tn)
        metrics["false_negative_rate"] = fn / max(1, fn + tp)

        return metrics

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            return {}

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_[0])
        else:
            return {}

        # Create feature importance dictionary
        feature_importance = {}
        for i, feature_name in enumerate(self.feature_names):
            feature_importance[feature_name] = float(importance[i])

        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return feature_importance

    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform hyperparameter tuning using GridSearchCV."""
        self.logger.info("Starting hyperparameter tuning")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Define parameter grids
        if self.model_type == "random_forest":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
        elif self.model_type == "xgboost":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
            }
        else:
            raise ValueError(
                f"Hyperparameter tuning not supported for {self.model_type}"
            )

        # Create base model
        base_model = self._create_model()

        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=self.cv_folds,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_scaled, y)

        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.model_params = grid_search.best_params_

        results = {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_,
        }

        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.3f}")

        return results

    def save_model(self) -> bool:
        """Save the trained model and related files."""
        try:
            # Save model
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)

            # Save scaler
            with open(self.scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)

            # Save feature names
            with open(self.feature_names_path, "w") as f:
                json.dump(self.feature_names, f)

            # Save model metadata
            metadata_path = self.model_path.replace(".pkl", "_metadata.json")
            metadata = {
                "model_type": self.model_type,
                "model_params": self.model_params,
                "feature_names": self.feature_names,
                "training_date": datetime.utcnow().isoformat(),
                "model_path": self.model_path,
                "scaler_path": self.scaler_path,
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Model saved to {self.model_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False

    def load_model(self) -> bool:
        """Load a trained model."""
        try:
            # Load model
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

            # Load scaler
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

            # Load feature names
            with open(self.feature_names_path, "r") as f:
                self.feature_names = json.load(f)

            self.logger.info(f"Model loaded from {self.model_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Make predictions
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)[:, 1]

        return predictions, probabilities

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if self.model is None:
            return {"status": "no_model"}

        return {
            "model_type": self.model_type,
            "model_params": self.model_params,
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names),
            "model_path": self.model_path,
        }


class ModelTrainingPipeline:
    """Complete pipeline for training fraud detection models."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the training pipeline."""
        self.config = config or {}
        self.trainer = FraudDetectionModelTrainer(config)
        self.logger = logging.getLogger(__name__)

    def train_from_transactions(
        self,
        transactions: List[Transaction],
        feature_engine: FeatureEngine,
        tune_hyperparameters: bool = False,
    ) -> Dict[str, Any]:
        """Complete training pipeline from transaction data."""
        self.logger.info("Starting model training pipeline")

        # Prepare data
        X, y = self.trainer.prepare_training_data(transactions, feature_engine)

        # Hyperparameter tuning (optional)
        if tune_hyperparameters:
            tuning_results = self.trainer.hyperparameter_tuning(X, y)
            self.logger.info("Hyperparameter tuning completed")

        # Train model
        training_results = self.trainer.train_model(X, y)

        # Save model
        save_success = self.trainer.save_model()

        # Combine results
        results = {
            "training_results": training_results,
            "model_saved": save_success,
            "model_info": self.trainer.get_model_info(),
        }

        if tune_hyperparameters:
            results["tuning_results"] = tuning_results

        self.logger.info("Model training pipeline completed")
        return results
