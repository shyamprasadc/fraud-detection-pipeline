#!/usr/bin/env python3
"""
Script to train the fraud detection model.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import redis
from src.ingestion.data_simulator import TransactionSimulator
from src.processing.feature_engine import FeatureEngine
from src.ml.model_trainer import ModelTrainingPipeline


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/train_model.log")],
    )


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument(
        "--transactions",
        type=int,
        default=10000,
        help="Number of transactions to generate for training",
    )
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.05,
        help="Fraud rate in training data (0.0-1.0)",
    )
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "xgboost"],
        default="random_forest",
        help="Model type to train",
    )
    parser.add_argument(
        "--tune-hyperparameters",
        action="store_true",
        help="Perform hyperparameter tuning",
    )
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument(
        "--output-dir", default="models", help="Output directory for trained model"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    try:
        logger.info("Starting model training process")

        # Initialize Redis connection
        logger.info("Connecting to Redis")
        redis_client = redis.Redis(
            host=args.redis_host, port=args.redis_port, db=0, decode_responses=True
        )

        # Test Redis connection
        redis_client.ping()
        logger.info("Redis connection established")

        # Initialize components
        logger.info("Initializing components")

        # Data simulator
        simulator_config = {"fraud_rate": args.fraud_rate}
        simulator = TransactionSimulator(simulator_config)

        # Feature engine
        feature_engine_config = {"cache_ttl": 3600}
        feature_engine = FeatureEngine(redis_client, feature_engine_config)

        # Model training pipeline
        model_config = {
            "model_type": args.model_type,
            "model_path": os.path.join(args.output_dir, "fraud_detection_model.pkl"),
            "scaler_path": os.path.join(args.output_dir, "scaler.pkl"),
            "feature_names_path": os.path.join(args.output_dir, "feature_names.json"),
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 5,
        }

        training_pipeline = ModelTrainingPipeline(model_config)

        # Generate training data
        logger.info(f"Generating {args.transactions} transactions for training")
        transactions = simulator.generate_transactions(
            count=args.transactions, fraud_rate=args.fraud_rate
        )

        logger.info(f"Generated {len(transactions)} transactions")
        logger.info(
            f"Fraud rate: {sum(1 for t in transactions if t.is_fraud) / len(transactions):.3f}"
        )

        # Train model
        logger.info("Starting model training")
        results = training_pipeline.train_from_transactions(
            transactions=transactions,
            feature_engine=feature_engine,
            tune_hyperparameters=args.tune_hyperparameters,
        )

        # Print results
        logger.info("Training completed successfully!")
        logger.info("=" * 50)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 50)

        training_results = results["training_results"]
        logger.info(f"Model Type: {training_results['model_type']}")
        logger.info(f"Training Accuracy: {training_results['train_score']:.3f}")
        logger.info(f"Test Accuracy: {training_results['test_score']:.3f}")
        logger.info(
            f"Cross-validation Score: {training_results['cv_mean']:.3f} (+/- {training_results['cv_std'] * 2:.3f})"
        )

        metrics = training_results["metrics"]
        logger.info(f"Precision: {metrics['precision']:.3f}")
        logger.info(f"Recall: {metrics['recall']:.3f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.3f}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.3f}")
        logger.info(f"Average Precision: {metrics['average_precision']:.3f}")

        logger.info(f"False Positive Rate: {metrics['false_positive_rate']:.3f}")
        logger.info(f"False Negative Rate: {metrics['false_negative_rate']:.3f}")

        # Feature importance
        logger.info("\nTop 10 Most Important Features:")
        feature_importance = training_results["feature_importance"]
        for i, (feature, importance) in enumerate(
            list(feature_importance.items())[:10]
        ):
            logger.info(f"{i + 1:2d}. {feature}: {importance:.4f}")

        # Model info
        model_info = results["model_info"]
        logger.info(f"\nModel saved to: {model_info['model_path']}")
        logger.info(f"Number of features: {model_info['n_features']}")

        if args.tune_hyperparameters and "tuning_results" in results:
            tuning_results = results["tuning_results"]
            logger.info(f"\nBest hyperparameters: {tuning_results['best_params']}")
            logger.info(f"Best CV score: {tuning_results['best_score']:.3f}")

        logger.info("=" * 50)
        logger.info("Model training completed successfully!")

        return 0

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
