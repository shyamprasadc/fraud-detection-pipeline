"""
Transaction data model for fraud detection pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Any
import uuid


@dataclass
class Location:
    """Geographic location data."""

    latitude: float
    longitude: float
    city: Optional[str] = None
    country: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "city": self.city,
            "country": self.country,
        }


@dataclass
class Transaction:
    """Transaction data model for fraud detection."""

    # Core transaction data
    transaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    merchant_id: str = ""
    amount: float = 0.0
    currency: str = "USD"
    transaction_type: str = "credit_card"  # credit_card, debit_card, digital_wallet

    # Merchant information
    merchant_category: str = ""
    merchant_name: str = ""

    # Location data
    location: Optional[Location] = None

    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Card information (for card transactions)
    card_last_four: Optional[str] = None
    card_type: Optional[str] = None  # visa, mastercard, amex, etc.

    # Device information
    device_id: Optional[str] = None
    ip_address: Optional[str] = None

    # Fraud detection fields (for training data)
    is_fraud: Optional[bool] = None
    fraud_type: Optional[str] = None  # stolen_card, account_takeover, etc.

    # Processing metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate transaction data after initialization."""
        if self.amount < 0:
            raise ValueError("Transaction amount cannot be negative")

        if self.transaction_type not in ["credit_card", "debit_card", "digital_wallet"]:
            raise ValueError(f"Invalid transaction type: {self.transaction_type}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary for serialization."""
        return {
            "transaction_id": self.transaction_id,
            "user_id": self.user_id,
            "merchant_id": self.merchant_id,
            "amount": self.amount,
            "currency": self.currency,
            "transaction_type": self.transaction_type,
            "merchant_category": self.merchant_category,
            "merchant_name": self.merchant_name,
            "location": self.location.to_dict() if self.location else None,
            "timestamp": self.timestamp.isoformat(),
            "card_last_four": self.card_last_four,
            "card_type": self.card_type,
            "device_id": self.device_id,
            "ip_address": self.ip_address,
            "is_fraud": self.is_fraud,
            "fraud_type": self.fraud_type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transaction":
        """Create transaction from dictionary."""
        # Handle location data
        location_data = data.get("location")
        location = None
        if location_data:
            location = Location(**location_data)

        # Handle timestamp conversion
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))

        return cls(
            transaction_id=data.get("transaction_id", str(uuid.uuid4())),
            user_id=data.get("user_id", ""),
            merchant_id=data.get("merchant_id", ""),
            amount=data.get("amount", 0.0),
            currency=data.get("currency", "USD"),
            transaction_type=data.get("transaction_type", "credit_card"),
            merchant_category=data.get("merchant_category", ""),
            merchant_name=data.get("merchant_name", ""),
            location=location,
            timestamp=timestamp or datetime.utcnow(),
            card_last_four=data.get("card_last_four"),
            card_type=data.get("card_type"),
            device_id=data.get("device_id"),
            ip_address=data.get("ip_address"),
            is_fraud=data.get("is_fraud"),
            fraud_type=data.get("fraud_type"),
            created_at=created_at or datetime.utcnow(),
            updated_at=updated_at or datetime.utcnow(),
        )

    def get_key_fields(self) -> Dict[str, Any]:
        """Get key fields for caching and indexing."""
        return {
            "transaction_id": self.transaction_id,
            "user_id": self.user_id,
            "merchant_id": self.merchant_id,
            "timestamp": self.timestamp,
        }
