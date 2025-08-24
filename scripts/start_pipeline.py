#!/usr/bin/env python3
"""
Script to start the complete fraud detection pipeline.
"""

import os
import sys
import argparse
import logging
import signal
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import redis
from src.ingestion.transaction_producer import (
    TransactionProducer,
    TransactionProducerConfig,
)
from src.processing.transaction_processor import (
    TransactionProcessor,
    TransactionProcessorConfig,
)
from src.alerting.alert_manager import AlertManager
from src.models.fraud_score import FraudScore


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/pipeline.log")],
    )


class FraudDetectionPipeline:
    """Main fraud detection pipeline orchestrator."""

    def __init__(self, config):
        """Initialize the pipeline."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Components
        self.redis_client = None
        self.producer = None
        self.processor = None
        self.alert_manager = None

        # Control flags
        self.running = False
        self.shutdown_requested = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_requested = True
        self.stop()

    def initialize(self):
        """Initialize all pipeline components."""
        try:
            self.logger.info("Initializing fraud detection pipeline")

            # Initialize Redis
            self.logger.info("Connecting to Redis")
            self.redis_client = redis.Redis(
                host=self.config.get("redis_host", "localhost"),
                port=self.config.get("redis_port", 6379),
                db=self.config.get("redis_db", 0),
                decode_responses=True,
            )
            self.redis_client.ping()
            self.logger.info("Redis connection established")

            # Initialize alert manager
            self.logger.info("Initializing alert manager")
            alert_config = {
                "redis_host": self.config.get("redis_host", "localhost"),
                "redis_port": self.config.get("redis_port", 6379),
                "redis_db": self.config.get("redis_db", 0),
                "notification_channels": self.config.get("notification_channels", []),
            }
            self.alert_manager = AlertManager(alert_config)

            # Initialize transaction processor
            self.logger.info("Initializing transaction processor")
            processor_config = TransactionProcessorConfig(
                kafka_bootstrap_servers=self.config.get(
                    "kafka_bootstrap_servers", "localhost:9092"
                ),
                kafka_topic_transactions=self.config.get(
                    "kafka_topic_transactions", "payment-transactions"
                ),
                kafka_consumer_group=self.config.get(
                    "kafka_consumer_group", "fraud-detection-group"
                ),
                max_workers=self.config.get("max_workers", 4),
                batch_size=self.config.get("batch_size", 100),
                batch_timeout=self.config.get("batch_timeout", 5.0),
            )

            self.processor = TransactionProcessor(processor_config.to_dict())

            # Initialize processor components
            model_config = {
                "model_path": self.config.get(
                    "model_path", "models/fraud_detection_model.pkl"
                ),
                "model_version": self.config.get("model_version", "v1.0"),
            }

            success = self.processor.initialize(self.redis_client, model_config)
            if not success:
                raise RuntimeError("Failed to initialize transaction processor")

            # Set callbacks
            self.processor.set_callbacks(
                fraud_score_callback=self._on_fraud_score, alert_callback=self._on_alert
            )

            # Initialize producer (optional)
            if self.config.get("start_producer", False):
                self.logger.info("Initializing transaction producer")
                producer_config = TransactionProducerConfig(
                    kafka_bootstrap_servers=self.config.get(
                        "kafka_bootstrap_servers", "localhost:9092"
                    ),
                    kafka_topic=self.config.get(
                        "kafka_topic_transactions", "payment-transactions"
                    ),
                )
                self.producer = TransactionProducer(producer_config.to_dict())

            self.logger.info("Pipeline initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing pipeline: {e}")
            return False

    def _on_fraud_score(self, fraud_score: FraudScore):
        """Callback for fraud score generation."""
        try:
            # Store fraud score in Redis
            score_key = f"fraud_score:{fraud_score.transaction_id}"
            self.redis_client.hset(score_key, mapping=fraud_score.to_dict())
            self.redis_client.expire(score_key, 86400)  # 24 hours TTL

            # Add to recent fraud scores for metrics
            self.redis_client.zadd(
                "recent_fraud_scores",
                {fraud_score.transaction_id: fraud_score.fraud_probability},
            )
            self.redis_client.expire("recent_fraud_scores", 3600)  # 1 hour TTL

            # Log high-risk transactions
            if fraud_score.risk_level.value in ["HIGH", "CRITICAL"]:
                self.logger.warning(
                    f"High-risk transaction: {fraud_score.transaction_id}, "
                    f"risk: {fraud_score.risk_level.value}, "
                    f"probability: {fraud_score.fraud_probability:.3f}"
                )

        except Exception as e:
            self.logger.error(f"Error in fraud score callback: {e}")

    def _on_alert(self, fraud_score: FraudScore):
        """Callback for alert generation."""
        try:
            # Create alert
            alert = self.alert_manager.create_alert(fraud_score)
            if alert:
                self.logger.info(
                    f"Alert created: {alert.alert_id} for transaction {fraud_score.transaction_id}"
                )

                # Store alert reference
                self.redis_client.set(
                    f"transaction_alert:{fraud_score.transaction_id}",
                    alert.alert_id,
                    ex=86400,
                )

        except Exception as e:
            self.logger.error(f"Error in alert callback: {e}")

    def start(self):
        """Start the pipeline."""
        if not self.running:
            self.logger.info("Starting fraud detection pipeline")
            self.running = True

            # Start producer in background if enabled
            if self.producer and self.config.get("start_producer", False):
                self._start_producer_background()

            # Start transaction processing
            self._start_processing()

    def _start_producer_background(self):
        """Start transaction producer in background."""
        import threading

        def producer_worker():
            try:
                tps = self.config.get("producer_tps", 10)
                duration_hours = self.config.get("producer_duration_hours", 1)
                fraud_rate = self.config.get("producer_fraud_rate", 0.05)

                self.logger.info(
                    f"Starting transaction producer: {tps} TPS for {duration_hours} hours"
                )
                self.producer.stream_transactions(tps, duration_hours, fraud_rate)

            except Exception as e:
                self.logger.error(f"Error in producer worker: {e}")

        producer_thread = threading.Thread(target=producer_worker, daemon=True)
        producer_thread.start()
        self.logger.info("Transaction producer started in background")

    def _start_processing(self):
        """Start transaction processing."""
        try:
            self.logger.info("Starting transaction processing")
            self.processor.start_consuming()
        except Exception as e:
            self.logger.error(f"Error in transaction processing: {e}")
        finally:
            self.running = False

    def stop(self):
        """Stop the pipeline."""
        if self.running:
            self.logger.info("Stopping fraud detection pipeline")
            self.running = False

            # Stop processor
            if self.processor:
                self.processor.stop_consuming()

            # Stop producer
            if self.producer:
                self.producer.close()

            self.logger.info("Pipeline stopped")

    def get_metrics(self):
        """Get pipeline metrics."""
        metrics = {
            "pipeline_status": "running" if self.running else "stopped",
            "shutdown_requested": self.shutdown_requested,
        }

        if self.processor:
            metrics.update(self.processor.get_metrics())

        if self.alert_manager:
            metrics["alert_statistics"] = self.alert_manager.get_alert_statistics()

        return metrics


def main():
    """Main pipeline script."""
    parser = argparse.ArgumentParser(description="Start fraud detection pipeline")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument(
        "--kafka-bootstrap-servers",
        default="localhost:9092",
        help="Kafka bootstrap servers",
    )
    parser.add_argument(
        "--kafka-topic",
        default="payment-transactions",
        help="Kafka topic for transactions",
    )
    parser.add_argument(
        "--consumer-group", default="fraud-detection-group", help="Kafka consumer group"
    )
    parser.add_argument(
        "--max-workers", type=int, default=4, help="Maximum number of worker threads"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for processing"
    )
    parser.add_argument(
        "--batch-timeout", type=float, default=5.0, help="Batch timeout in seconds"
    )
    parser.add_argument(
        "--model-path",
        default="models/fraud_detection_model.pkl",
        help="Path to trained model",
    )
    parser.add_argument(
        "--start-producer", action="store_true", help="Start transaction producer"
    )
    parser.add_argument(
        "--producer-tps",
        type=int,
        default=10,
        help="Transactions per second for producer",
    )
    parser.add_argument(
        "--producer-duration-hours",
        type=int,
        default=1,
        help="Duration to run producer in hours",
    )
    parser.add_argument(
        "--producer-fraud-rate",
        type=float,
        default=0.05,
        help="Fraud rate for producer",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Configuration
    config = {
        "redis_host": args.redis_host,
        "redis_port": args.redis_port,
        "redis_db": 0,
        "kafka_bootstrap_servers": args.kafka_bootstrap_servers,
        "kafka_topic_transactions": args.kafka_topic,
        "kafka_consumer_group": args.consumer_group,
        "max_workers": args.max_workers,
        "batch_size": args.batch_size,
        "batch_timeout": args.batch_timeout,
        "model_path": args.model_path,
        "model_version": "v1.0",
        "start_producer": args.start_producer,
        "producer_tps": args.producer_tps,
        "producer_duration_hours": args.producer_duration_hours,
        "producer_fraud_rate": args.producer_fraud_rate,
        "notification_channels": [],
    }

    try:
        # Create and initialize pipeline
        pipeline = FraudDetectionPipeline(config)

        if not pipeline.initialize():
            logger.error("Failed to initialize pipeline")
            return 1

        # Start pipeline
        pipeline.start()

        # Keep running until shutdown
        while pipeline.running and not pipeline.shutdown_requested:
            time.sleep(1)

            # Print metrics every 30 seconds
            if int(time.time()) % 30 == 0:
                metrics = pipeline.get_metrics()
                logger.info(f"Pipeline metrics: {metrics}")

        return 0

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
