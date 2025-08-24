"""
Kafka producer for streaming transaction data.
"""

import json
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from kafka import KafkaProducer
from kafka.errors import KafkaError, KafkaTimeoutError

from ..models.transaction import Transaction
from .data_simulator import TransactionSimulator


class TransactionProducer:
    """Kafka producer for streaming transaction data."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the transaction producer."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Kafka configuration
        self.bootstrap_servers = config.get("kafka_bootstrap_servers", "localhost:9092")
        self.topic = config.get("kafka_topic", "payment-transactions")
        self.producer_config = config.get("producer_config", {})

        # Initialize Kafka producer
        self.producer = self._create_producer()

        # Initialize data simulator
        self.simulator = TransactionSimulator(config.get("simulator_config", {}))

        # Metrics
        self.messages_sent = 0
        self.messages_failed = 0
        self.start_time = datetime.utcnow()

    def _create_producer(self) -> KafkaProducer:
        """Create and configure Kafka producer."""
        try:
            producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",  # Wait for all replicas
                retries=3,  # Retry failed sends
                **self.producer_config,
            )
            self.logger.info(
                f"Kafka producer created successfully for topic: {self.topic}"
            )
            return producer
        except Exception as e:
            self.logger.error(f"Failed to create Kafka producer: {e}")
            raise

    def send_transaction(
        self, transaction: Transaction, key: Optional[str] = None
    ) -> bool:
        """Send a single transaction to Kafka."""
        try:
            # Use user_id as key for partitioning if not specified
            if key is None:
                key = transaction.user_id

            # Convert transaction to dictionary
            message = transaction.to_dict()

            # Send to Kafka
            future = self.producer.send(topic=self.topic, key=key, value=message)

            # Wait for send to complete
            record_metadata = future.get(timeout=10)

            self.messages_sent += 1
            self.logger.debug(
                f"Transaction sent successfully: {transaction.transaction_id} "
                f"to partition {record_metadata.partition} "
                f"at offset {record_metadata.offset}"
            )

            return True

        except KafkaTimeoutError as e:
            self.messages_failed += 1
            self.logger.error(
                f"Kafka timeout error sending transaction {transaction.transaction_id}: {e}"
            )
            return False
        except KafkaError as e:
            self.messages_failed += 1
            self.logger.error(
                f"Kafka error sending transaction {transaction.transaction_id}: {e}"
            )
            return False
        except Exception as e:
            self.messages_failed += 1
            self.logger.error(
                f"Unexpected error sending transaction {transaction.transaction_id}: {e}"
            )
            return False

    def send_transactions_batch(
        self, transactions: List[Transaction]
    ) -> Dict[str, int]:
        """Send a batch of transactions to Kafka."""
        results = {"sent": 0, "failed": 0}

        for transaction in transactions:
            success = self.send_transaction(transaction)
            if success:
                results["sent"] += 1
            else:
                results["failed"] += 1

        self.logger.info(
            f"Batch sent: {results['sent']} successful, {results['failed']} failed"
        )
        return results

    def stream_transactions(
        self, tps: int = 10, duration_hours: int = 1, fraud_rate: float = 0.05
    ):
        """Stream transactions at specified rate."""
        self.logger.info(
            f"Starting transaction stream: {tps} TPS for {duration_hours} hours"
        )

        start_time = time.time()
        end_time = start_time + (duration_hours * 3600)

        # Calculate delay between transactions
        delay = 1.0 / tps

        transaction_count = 0

        try:
            for transaction in self.simulator.stream_transactions(
                tps, duration_hours, fraud_rate
            ):
                if time.time() > end_time:
                    break

                # Send transaction
                success = self.send_transaction(transaction)
                transaction_count += 1

                # Log progress
                if transaction_count % 100 == 0:
                    elapsed = time.time() - start_time
                    actual_tps = transaction_count / elapsed
                    self.logger.info(
                        f"Progress: {transaction_count} transactions sent, "
                        f"actual TPS: {actual_tps:.2f}, "
                        f"success rate: {self.messages_sent / (self.messages_sent + self.messages_failed) * 100:.1f}%"
                    )

                # Rate limiting
                time.sleep(delay)

        except KeyboardInterrupt:
            self.logger.info("Transaction stream interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in transaction stream: {e}")
        finally:
            self.logger.info(
                f"Transaction stream completed: {transaction_count} transactions sent"
            )

    def generate_and_send_batch(
        self, count: int, fraud_rate: float = 0.05
    ) -> Dict[str, int]:
        """Generate and send a batch of transactions."""
        self.logger.info(f"Generating and sending batch of {count} transactions")

        # Generate transactions
        transactions = self.simulator.generate_transactions(count, fraud_rate)

        # Send batch
        results = self.send_transactions_batch(transactions)

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get producer metrics."""
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()

        return {
            "messages_sent": self.messages_sent,
            "messages_failed": self.messages_failed,
            "success_rate": self.messages_sent
            / max(1, self.messages_sent + self.messages_failed),
            "messages_per_second": self.messages_sent / max(1, elapsed),
            "uptime_seconds": elapsed,
            "start_time": self.start_time.isoformat(),
        }

    def close(self):
        """Close the Kafka producer."""
        if self.producer:
            self.producer.flush()  # Wait for all messages to be sent
            self.producer.close()
            self.logger.info("Kafka producer closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class TransactionProducerConfig:
    """Configuration for transaction producer."""

    def __init__(self, **kwargs):
        """Initialize configuration."""
        self.kafka_bootstrap_servers = kwargs.get(
            "kafka_bootstrap_servers", "localhost:9092"
        )
        self.kafka_topic = kwargs.get("kafka_topic", "payment-transactions")
        self.producer_config = kwargs.get("producer_config", {})
        self.simulator_config = kwargs.get("simulator_config", {})

        # Default producer settings
        if not self.producer_config:
            self.producer_config = {
                "acks": "all",
                "retries": 3,
                "batch_size": 16384,
                "linger_ms": 10,
                "buffer_memory": 33554432,
                "compression_type": "gzip",
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "kafka_bootstrap_servers": self.kafka_bootstrap_servers,
            "kafka_topic": self.kafka_topic,
            "producer_config": self.producer_config,
            "simulator_config": self.simulator_config,
        }
