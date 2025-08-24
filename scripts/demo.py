#!/usr/bin/env python3
"""
Demo script for the fraud detection pipeline.
This script demonstrates the complete pipeline from data generation to fraud detection.
"""

import os
import sys
import time
import logging
import threading
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import redis
from src.ingestion.data_simulator import TransactionSimulator
from src.ingestion.transaction_producer import TransactionProducer
from src.processing.feature_engine import FeatureEngine
from src.ml.model_trainer import ModelTrainingPipeline
from src.processing.fraud_detector import FraudDetectionPipeline
from src.alerting.alert_manager import AlertManager
from src.models.transaction import Transaction
from src.models.fraud_score import FraudScore


class FraudDetectionDemo:
    """Demo class for showcasing the fraud detection pipeline."""

    def __init__(self):
        """Initialize the demo."""
        self.logger = self._setup_logging()
        self.redis_client = None
        self.running = False

        # Demo components
        self.simulator = None
        self.producer = None
        self.feature_engine = None
        self.fraud_pipeline = None
        self.alert_manager = None

        # Demo metrics
        self.metrics = {
            "transactions_generated": 0,
            "transactions_processed": 0,
            "fraud_scores_generated": 0,
            "alerts_created": 0,
            "start_time": None,
        }

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler("logs/demo.log")],
        )
        return logging.getLogger(__name__)

    def initialize(self):
        """Initialize all demo components."""
        self.logger.info("🚀 Initializing Fraud Detection Demo")

        try:
            # Initialize Redis
            self.logger.info("Connecting to Redis...")
            self.redis_client = redis.Redis(
                host="localhost", port=6379, db=0, decode_responses=True
            )
            self.redis_client.ping()
            self.logger.info("✅ Redis connection established")

            # Initialize components
            self.logger.info("Initializing components...")

            # Data simulator
            self.simulator = TransactionSimulator()
            self.logger.info("✅ Data simulator initialized")

            # Feature engine
            self.feature_engine = FeatureEngine(self.redis_client)
            self.logger.info("✅ Feature engine initialized")

            # Alert manager
            self.alert_manager = AlertManager(
                {"redis_host": "localhost", "redis_port": 6379, "redis_db": 0}
            )
            self.logger.info("✅ Alert manager initialized")

            # Fraud detection pipeline
            self.fraud_pipeline = FraudDetectionPipeline(
                self.feature_engine,
                None,  # Will be set after model training
                {},
            )
            self.logger.info("✅ Fraud detection pipeline initialized")

            self.logger.info("🎉 Demo initialization completed successfully!")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error initializing demo: {e}")
            return False

    def train_model(self):
        """Train a fraud detection model."""
        self.logger.info("🤖 Training fraud detection model...")

        try:
            # Generate training data
            self.logger.info("Generating training data...")
            training_transactions = self.simulator.generate_transactions(
                count=5000, fraud_rate=0.05
            )
            self.logger.info(
                f"Generated {len(training_transactions)} training transactions"
            )

            # Train model
            model_config = {
                "model_type": "random_forest",
                "model_path": "models/fraud_detection_model.pkl",
                "test_size": 0.2,
                "random_state": 42,
            }

            training_pipeline = ModelTrainingPipeline(model_config)
            results = training_pipeline.train_from_transactions(
                transactions=training_transactions,
                feature_engine=self.feature_engine,
                tune_hyperparameters=False,
            )

            # Update fraud pipeline with trained model
            self.fraud_pipeline.model_trainer = training_pipeline.trainer

            self.logger.info("✅ Model training completed successfully!")
            self.logger.info(
                f"Model accuracy: {results['training_results']['test_score']:.3f}"
            )

            return True

        except Exception as e:
            self.logger.error(f"❌ Error training model: {e}")
            return False

    def run_demo(self, duration_minutes=5, tps=10):
        """Run the complete demo pipeline."""
        self.logger.info(
            f"🎬 Starting demo pipeline ({duration_minutes} minutes, {tps} TPS)"
        )

        self.running = True
        self.metrics["start_time"] = time.time()

        try:
            # Start transaction generation in background
            producer_thread = threading.Thread(
                target=self._generate_transactions,
                args=(duration_minutes, tps),
                daemon=True,
            )
            producer_thread.start()

            # Start transaction processing in background
            processor_thread = threading.Thread(
                target=self._process_transactions, daemon=True
            )
            processor_thread.start()

            # Start metrics monitoring
            self._monitor_metrics(duration_minutes)

        except KeyboardInterrupt:
            self.logger.info("Demo interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in demo: {e}")
        finally:
            self.stop_demo()

    def _generate_transactions(self, duration_minutes, tps):
        """Generate transactions in background."""
        try:
            end_time = time.time() + (duration_minutes * 60)
            delay = 1.0 / tps

            while self.running and time.time() < end_time:
                # Generate transaction
                transaction = self.simulator.generate_transaction()

                # Store in Redis for processing
                transaction_key = f"demo_transaction:{transaction.transaction_id}"
                self.redis_client.hset(transaction_key, mapping=transaction.to_dict())
                self.redis_client.expire(transaction_key, 3600)  # 1 hour TTL

                # Add to processing queue
                self.redis_client.lpush(
                    "demo_transaction_queue", transaction.transaction_id
                )

                self.metrics["transactions_generated"] += 1

                # Rate limiting
                time.sleep(delay)

        except Exception as e:
            self.logger.error(f"Error generating transactions: {e}")

    def _process_transactions(self):
        """Process transactions in background."""
        try:
            while self.running:
                # Get transaction from queue
                transaction_id = self.redis_client.brpop(
                    "demo_transaction_queue", timeout=1
                )

                if transaction_id:
                    transaction_id = transaction_id[1]

                    # Get transaction data
                    transaction_data = self.redis_client.hgetall(
                        f"demo_transaction:{transaction_id}"
                    )
                    if transaction_data:
                        # Convert to Transaction object
                        transaction = Transaction.from_dict(transaction_data)

                        # Process through fraud detection pipeline
                        fraud_score = self.fraud_pipeline.process_transaction(
                            transaction
                        )

                        # Store fraud score
                        score_key = f"demo_fraud_score:{transaction_id}"
                        self.redis_client.hset(score_key, mapping=fraud_score.to_dict())
                        self.redis_client.expire(score_key, 3600)

                        # Create alert if needed
                        if self.fraud_pipeline.fraud_detector.should_trigger_alert(
                            fraud_score
                        ):
                            alert = self.alert_manager.create_alert(fraud_score)
                            if alert:
                                self.metrics["alerts_created"] += 1

                        self.metrics["transactions_processed"] += 1
                        self.metrics["fraud_scores_generated"] += 1

        except Exception as e:
            self.logger.error(f"Error processing transactions: {e}")

    def _monitor_metrics(self, duration_minutes):
        """Monitor and display metrics."""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        while self.running and time.time() < end_time:
            elapsed = time.time() - start_time

            # Calculate metrics
            tps_generated = self.metrics["transactions_generated"] / max(elapsed, 1)
            tps_processed = self.metrics["transactions_processed"] / max(elapsed, 1)

            # Display metrics
            self.logger.info("=" * 60)
            self.logger.info("📊 DEMO METRICS")
            self.logger.info("=" * 60)
            self.logger.info(f"Elapsed Time: {elapsed:.1f}s")
            self.logger.info(
                f"Transactions Generated: {self.metrics['transactions_generated']}"
            )
            self.logger.info(
                f"Transactions Processed: {self.metrics['transactions_processed']}"
            )
            self.logger.info(
                f"Fraud Scores Generated: {self.metrics['fraud_scores_generated']}"
            )
            self.logger.info(f"Alerts Created: {self.metrics['alerts_created']}")
            self.logger.info(f"Generation Rate: {tps_generated:.2f} TPS")
            self.logger.info(f"Processing Rate: {tps_processed:.2f} TPS")
            self.logger.info("=" * 60)

            time.sleep(10)  # Update every 10 seconds

    def stop_demo(self):
        """Stop the demo."""
        self.logger.info("🛑 Stopping demo...")
        self.running = False

        # Print final metrics
        if self.metrics["start_time"]:
            total_time = time.time() - self.metrics["start_time"]
            self.logger.info("=" * 60)
            self.logger.info("🏁 FINAL DEMO RESULTS")
            self.logger.info("=" * 60)
            self.logger.info(f"Total Runtime: {total_time:.1f}s")
            self.logger.info(
                f"Total Transactions Generated: {self.metrics['transactions_generated']}"
            )
            self.logger.info(
                f"Total Transactions Processed: {self.metrics['transactions_processed']}"
            )
            self.logger.info(
                f"Total Fraud Scores: {self.metrics['fraud_scores_generated']}"
            )
            self.logger.info(f"Total Alerts Created: {self.metrics['alerts_created']}")
            self.logger.info(
                f"Average Generation Rate: {self.metrics['transactions_generated'] / max(total_time, 1):.2f} TPS"
            )
            self.logger.info(
                f"Average Processing Rate: {self.metrics['transactions_processed'] / max(total_time, 1):.2f} TPS"
            )
            self.logger.info("=" * 60)

    def show_demo_info(self):
        """Show demo information."""
        self.logger.info("🎯 FRAUD DETECTION PIPELINE DEMO")
        self.logger.info("=" * 60)
        self.logger.info("This demo showcases:")
        self.logger.info("✅ Real-time transaction generation")
        self.logger.info("✅ Feature engineering")
        self.logger.info("✅ ML-based fraud detection")
        self.logger.info("✅ Alert management")
        self.logger.info("✅ Performance monitoring")
        self.logger.info("=" * 60)
        self.logger.info("Components:")
        self.logger.info("• Data Simulator: Generates realistic transaction data")
        self.logger.info("• Feature Engine: Calculates real-time features")
        self.logger.info("• ML Model: Random Forest classifier for fraud detection")
        self.logger.info("• Fraud Detector: Combines ML and rule-based detection")
        self.logger.info("• Alert Manager: Creates and manages fraud alerts")
        self.logger.info("=" * 60)


def main():
    """Main demo function."""
    import argparse

    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline Demo")
    parser.add_argument(
        "--duration", type=int, default=5, help="Demo duration in minutes"
    )
    parser.add_argument("--tps", type=int, default=10, help="Transactions per second")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training (use existing model)",
    )

    args = parser.parse_args()

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Create demo instance
    demo = FraudDetectionDemo()

    # Show demo info
    demo.show_demo_info()

    # Initialize demo
    if not demo.initialize():
        print("❌ Failed to initialize demo")
        return 1

    # Train model (unless skipped)
    if not args.skip_training:
        if not demo.train_model():
            print("❌ Failed to train model")
            return 1
    else:
        print("⏭️  Skipping model training")

    # Run demo
    try:
        demo.run_demo(duration_minutes=args.duration, tps=args.tps)
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")

    return 0


if __name__ == "__main__":
    sys.exit(main())
