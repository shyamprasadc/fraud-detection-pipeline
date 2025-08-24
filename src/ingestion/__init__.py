"""
Data ingestion components for fraud detection pipeline.
"""

from .data_simulator import TransactionSimulator
from .transaction_producer import TransactionProducer, TransactionProducerConfig

__all__ = ["TransactionSimulator", "TransactionProducer", "TransactionProducerConfig"]
