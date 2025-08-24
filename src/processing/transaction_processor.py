"""
Transaction processor for orchestrating the fraud detection pipeline.
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from kafka import KafkaConsumer
from kafka.errors import KafkaError

from ..models.transaction import Transaction
from ..models.fraud_score import FraudScore
from ..processing.fraud_detector import FraudDetectionPipeline
from ..processing.feature_engine import FeatureEngine
from ..ml.model_trainer import FraudDetectionModelTrainer


class TransactionProcessor:
    """Main transaction processor for fraud detection pipeline."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the transaction processor."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Kafka configuration
        self.kafka_config = {
            "bootstrap_servers": config.get(
                "kafka_bootstrap_servers", "localhost:9092"
            ),
            "topic": config.get("kafka_topic_transactions", "payment-transactions"),
            "group_id": config.get("kafka_consumer_group", "fraud-detection-group"),
            "auto_offset_reset": "latest",
            "enable_auto_commit": True,
            "value_deserializer": lambda x: json.loads(x.decode("utf-8")),
        }

        # Processing configuration
        self.max_workers = config.get("max_workers", 4)
        self.batch_size = config.get("batch_size", 100)
        self.batch_timeout = config.get("batch_timeout", 5.0)  # seconds

        # Initialize components
        self.feature_engine = None
        self.model_trainer = None
        self.fraud_pipeline = None
        self.consumer = None

        # Callbacks
        self.fraud_score_callback = None
        self.alert_callback = None

        # Metrics
        self.metrics = {
            "transactions_processed": 0,
            "fraud_scores_generated": 0,
            "alerts_triggered": 0,
            "processing_errors": 0,
            "start_time": datetime.utcnow(),
            "last_processed_time": None,
        }

    def initialize(self, redis_client, model_config: Dict[str, Any] = None):
        """Initialize the processing pipeline."""
        try:
            # Initialize feature engine
            self.feature_engine = FeatureEngine(
                redis_client, self.config.get("feature_engine_config", {})
            )

            # Initialize model trainer
            model_config = model_config or self.config.get("model_config", {})
            self.model_trainer = FraudDetectionModelTrainer(model_config)

            # Initialize fraud detection pipeline
            self.fraud_pipeline = FraudDetectionPipeline(
                self.feature_engine,
                self.model_trainer,
                self.config.get("fraud_detector_config", {}),
            )

            # Initialize Kafka consumer
            self.consumer = KafkaConsumer(**self.kafka_config)
            self.consumer.subscribe([self.kafka_config["topic"]])

            self.logger.info("Transaction processor initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing transaction processor: {e}")
            return False

    def set_callbacks(
        self,
        fraud_score_callback: Callable[[FraudScore], None] = None,
        alert_callback: Callable[[FraudScore], None] = None,
    ):
        """Set callback functions for fraud scores and alerts."""
        self.fraud_score_callback = fraud_score_callback
        self.alert_callback = alert_callback

    def process_transaction(
        self, transaction_data: Dict[str, Any]
    ) -> Optional[FraudScore]:
        """Process a single transaction."""
        try:
            # Parse transaction
            transaction = Transaction.from_dict(transaction_data)

            # Process through fraud detection pipeline
            fraud_score = self.fraud_pipeline.process_transaction(transaction)

            # Update metrics
            self.metrics["transactions_processed"] += 1
            self.metrics["fraud_scores_generated"] += 1
            self.metrics["last_processed_time"] = datetime.utcnow()

            # Trigger callbacks
            if self.fraud_score_callback:
                self.fraud_score_callback(fraud_score)

            if (
                self.alert_callback
                and self.fraud_pipeline.fraud_detector.should_trigger_alert(fraud_score)
            ):
                self.alert_callback(fraud_score)
                self.metrics["alerts_triggered"] += 1

            return fraud_score

        except Exception as e:
            self.metrics["processing_errors"] += 1
            self.logger.error(f"Error processing transaction: {e}")
            return None

    def process_transactions_batch(
        self, transactions_data: List[Dict[str, Any]]
    ) -> List[FraudScore]:
        """Process a batch of transactions in parallel."""
        fraud_scores = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all transactions for processing
            future_to_transaction = {
                executor.submit(self.process_transaction, txn_data): txn_data
                for txn_data in transactions_data
            }

            # Collect results
            for future in as_completed(future_to_transaction):
                try:
                    fraud_score = future.result()
                    if fraud_score:
                        fraud_scores.append(fraud_score)
                except Exception as e:
                    self.logger.error(f"Error in batch processing: {e}")
                    self.metrics["processing_errors"] += 1

        return fraud_scores

    def start_consuming(self):
        """Start consuming transactions from Kafka."""
        if not self.consumer:
            raise RuntimeError("Transaction processor not initialized")

        self.logger.info(
            f"Starting to consume transactions from topic: {self.kafka_config['topic']}"
        )

        try:
            batch = []
            batch_start_time = time.time()

            for message in self.consumer:
                try:
                    # Add message to batch
                    batch.append(message.value)

                    # Process batch if size or timeout reached
                    current_time = time.time()
                    if len(batch) >= self.batch_size or (
                        len(batch) > 0
                        and current_time - batch_start_time >= self.batch_timeout
                    ):
                        # Process batch
                        fraud_scores = self.process_transactions_batch(batch)

                        # Log batch results
                        self.logger.info(
                            f"Processed batch of {len(batch)} transactions, "
                            f"generated {len(fraud_scores)} fraud scores"
                        )

                        # Reset batch
                        batch = []
                        batch_start_time = current_time

                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    self.metrics["processing_errors"] += 1
                    continue

        except KeyboardInterrupt:
            self.logger.info("Transaction consumption interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in transaction consumption: {e}")
        finally:
            self.stop_consuming()

    def stop_consuming(self):
        """Stop consuming transactions."""
        if self.consumer:
            self.consumer.close()
            self.logger.info("Transaction consumption stopped")

    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        uptime = (datetime.utcnow() - self.metrics["start_time"]).total_seconds()

        return {
            **self.metrics,
            "uptime_seconds": uptime,
            "transactions_per_second": self.metrics["transactions_processed"]
            / max(1, uptime),
            "error_rate": self.metrics["processing_errors"]
            / max(1, self.metrics["transactions_processed"]),
            "pipeline_metrics": self.fraud_pipeline.get_pipeline_metrics()
            if self.fraud_pipeline
            else {},
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the processor."""
        health_status = {
            "status": "healthy",
            "components": {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Check feature engine
        if self.feature_engine:
            health_status["components"]["feature_engine"] = "healthy"
        else:
            health_status["components"]["feature_engine"] = "not_initialized"
            health_status["status"] = "unhealthy"

        # Check model trainer
        if self.model_trainer:
            health_status["components"]["model_trainer"] = "healthy"
        else:
            health_status["components"]["model_trainer"] = "not_initialized"
            health_status["status"] = "unhealthy"

        # Check fraud pipeline
        if self.fraud_pipeline:
            health_status["components"]["fraud_pipeline"] = "healthy"
        else:
            health_status["components"]["fraud_pipeline"] = "not_initialized"
            health_status["status"] = "unhealthy"

        # Check Kafka consumer
        if self.consumer:
            health_status["components"]["kafka_consumer"] = "healthy"
        else:
            health_status["components"]["kafka_consumer"] = "not_initialized"
            health_status["status"] = "unhealthy"

        return health_status


class TransactionProcessorConfig:
    """Configuration for transaction processor."""

    def __init__(self, **kwargs):
        """Initialize configuration."""
        # Kafka settings
        self.kafka_bootstrap_servers = kwargs.get(
            "kafka_bootstrap_servers", "localhost:9092"
        )
        self.kafka_topic_transactions = kwargs.get(
            "kafka_topic_transactions", "payment-transactions"
        )
        self.kafka_consumer_group = kwargs.get(
            "kafka_consumer_group", "fraud-detection-group"
        )

        # Processing settings
        self.max_workers = kwargs.get("max_workers", 4)
        self.batch_size = kwargs.get("batch_size", 100)
        self.batch_timeout = kwargs.get("batch_timeout", 5.0)

        # Component configurations
        self.feature_engine_config = kwargs.get("feature_engine_config", {})
        self.model_config = kwargs.get("model_config", {})
        self.fraud_detector_config = kwargs.get("fraud_detector_config", {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "kafka_bootstrap_servers": self.kafka_bootstrap_servers,
            "kafka_topic_transactions": self.kafka_topic_transactions,
            "kafka_consumer_group": self.kafka_consumer_group,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "batch_timeout": self.batch_timeout,
            "feature_engine_config": self.feature_engine_config,
            "model_config": self.model_config,
            "fraud_detector_config": self.fraud_detector_config,
        }
