"""
Data simulator for generating realistic transaction data.
"""

import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Generator
import json
import uuid

from ..models.transaction import Transaction, Location


class TransactionSimulator:
    """Simulates realistic payment transactions for fraud detection testing."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the transaction simulator."""
        self.config = config or {}

        # Merchant data
        self.merchants = self._load_merchants()
        self.merchant_categories = list(set(m["category"] for m in self.merchants))

        # User profiles
        self.user_profiles = self._generate_user_profiles()

        # Geographic data
        self.cities = self._load_cities()

        # Fraud patterns
        self.fraud_patterns = self._define_fraud_patterns()

        # Simulation state
        self.current_time = datetime.utcnow()
        self.transaction_counter = 0

    def _load_merchants(self) -> List[Dict[str, Any]]:
        """Load merchant data with categories and risk scores."""
        return [
            {
                "id": "amazon",
                "name": "Amazon",
                "category": "online_retail",
                "risk_score": 0.1,
            },
            {
                "id": "starbucks",
                "name": "Starbucks",
                "category": "food_beverage",
                "risk_score": 0.05,
            },
            {
                "id": "uber",
                "name": "Uber",
                "category": "transportation",
                "risk_score": 0.15,
            },
            {
                "id": "netflix",
                "name": "Netflix",
                "category": "entertainment",
                "risk_score": 0.05,
            },
            {
                "id": "walmart",
                "name": "Walmart",
                "category": "retail",
                "risk_score": 0.08,
            },
            {
                "id": "gas_station",
                "name": "Shell Gas Station",
                "category": "gas_station",
                "risk_score": 0.12,
            },
            {
                "id": "restaurant",
                "name": "Local Restaurant",
                "category": "food_beverage",
                "risk_score": 0.07,
            },
            {
                "id": "electronics_store",
                "name": "Best Buy",
                "category": "electronics",
                "risk_score": 0.2,
            },
            {
                "id": "jewelry_store",
                "name": "Tiffany & Co",
                "category": "jewelry",
                "risk_score": 0.3,
            },
            {
                "id": "travel_agency",
                "name": "Expedia",
                "category": "travel",
                "risk_score": 0.25,
            },
            {"id": "gaming", "name": "Steam", "category": "gaming", "risk_score": 0.1},
            {
                "id": "pharmacy",
                "name": "CVS Pharmacy",
                "category": "healthcare",
                "risk_score": 0.06,
            },
            {
                "id": "suspicious_merchant",
                "name": "Unknown Online Store",
                "category": "online_retail",
                "risk_score": 0.8,
            },
            {
                "id": "international_merchant",
                "name": "International Shop",
                "category": "online_retail",
                "risk_score": 0.6,
            },
        ]

    def _load_cities(self) -> List[Dict[str, Any]]:
        """Load city data with coordinates."""
        return [
            {"name": "New York", "lat": 40.7128, "lon": -74.0060, "country": "US"},
            {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437, "country": "US"},
            {"name": "Chicago", "lat": 41.8781, "lon": -87.6298, "country": "US"},
            {"name": "Houston", "lat": 29.7604, "lon": -95.3698, "country": "US"},
            {"name": "Phoenix", "lat": 33.4484, "lon": -112.0740, "country": "US"},
            {"name": "Philadelphia", "lat": 39.9526, "lon": -75.1652, "country": "US"},
            {"name": "San Antonio", "lat": 29.4241, "lon": -98.4936, "country": "US"},
            {"name": "San Diego", "lat": 32.7157, "lon": -117.1611, "country": "US"},
            {"name": "Dallas", "lat": 32.7767, "lon": -96.7970, "country": "US"},
            {"name": "San Jose", "lat": 37.3382, "lon": -121.8863, "country": "US"},
            {"name": "London", "lat": 51.5074, "lon": -0.1278, "country": "UK"},
            {"name": "Paris", "lat": 48.8566, "lon": 2.3522, "country": "FR"},
            {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503, "country": "JP"},
            {"name": "Moscow", "lat": 55.7558, "lon": 37.6176, "country": "RU"},
        ]

    def _generate_user_profiles(self) -> List[Dict[str, Any]]:
        """Generate realistic user profiles."""
        profiles = []
        for i in range(100):  # Generate 100 user profiles
            user_id = f"user_{i:03d}"

            # Random home location
            home_city = random.choice(self.cities)

            # Spending patterns
            avg_amount = random.uniform(25, 200)
            daily_transactions = random.randint(1, 5)

            # Preferred merchants
            preferred_merchants = random.sample(
                [m["id"] for m in self.merchants if m["risk_score"] < 0.3],
                random.randint(3, 8),
            )

            profiles.append(
                {
                    "user_id": user_id,
                    "home_location": {
                        "latitude": home_city["lat"] + random.uniform(-0.1, 0.1),
                        "longitude": home_city["lon"] + random.uniform(-0.1, 0.1),
                        "city": home_city["name"],
                        "country": home_city["country"],
                    },
                    "avg_transaction_amount": avg_amount,
                    "daily_transaction_count": daily_transactions,
                    "preferred_merchants": preferred_merchants,
                    "account_created": datetime.utcnow()
                    - timedelta(days=random.randint(30, 365)),
                    "fraud_history": random.random() < 0.05,  # 5% have fraud history
                }
            )

        return profiles

    def _define_fraud_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Define different types of fraud patterns."""
        return {
            "stolen_card": {
                "frequency_multiplier": 10,  # 10x normal frequency
                "amount_multiplier": 3,  # 3x normal amounts
                "geographic_spread": 1000,  # 1000km from home
                "time_pattern": "random",  # Random times
                "merchant_preference": "high_risk",
            },
            "account_takeover": {
                "frequency_multiplier": 5,
                "amount_multiplier": 2,
                "geographic_spread": 500,
                "time_pattern": "business_hours",
                "merchant_preference": "online_retail",
            },
            "merchant_fraud": {
                "frequency_multiplier": 1,
                "amount_multiplier": 5,
                "geographic_spread": 50,
                "time_pattern": "normal",
                "merchant_preference": "specific_merchant",
            },
        }

    def generate_transaction(
        self, is_fraud: bool = False, fraud_type: str = None
    ) -> Transaction:
        """Generate a single transaction."""
        # Select user
        user_profile = random.choice(self.user_profiles)
        user_id = user_profile["user_id"]

        # Select merchant
        if is_fraud and fraud_type:
            merchant = self._select_merchant_for_fraud(fraud_type, user_profile)
        else:
            merchant = self._select_merchant_normal(user_profile)

        # Generate amount
        if is_fraud and fraud_type:
            amount = self._generate_fraud_amount(user_profile, fraud_type)
        else:
            amount = self._generate_normal_amount(user_profile)

        # Generate location
        if is_fraud and fraud_type:
            location = self._generate_fraud_location(user_profile, fraud_type)
        else:
            location = self._generate_normal_location(user_profile)

        # Generate timestamp
        timestamp = self._generate_timestamp(is_fraud, fraud_type)

        # Transaction type
        transaction_type = random.choice(
            ["credit_card", "debit_card", "digital_wallet"]
        )

        # Card information (for card transactions)
        card_last_four = None
        card_type = None
        if transaction_type in ["credit_card", "debit_card"]:
            card_last_four = f"{random.randint(1000, 9999)}"
            card_type = random.choice(["visa", "mastercard", "amex"])

        # Create transaction
        transaction = Transaction(
            transaction_id=str(uuid.uuid4()),
            user_id=user_id,
            merchant_id=merchant["id"],
            amount=amount,
            currency="USD",
            transaction_type=transaction_type,
            merchant_category=merchant["category"],
            merchant_name=merchant["name"],
            location=Location(**location),
            timestamp=timestamp,
            card_last_four=card_last_four,
            card_type=card_type,
            device_id=f"device_{random.randint(1000, 9999)}",
            ip_address=f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
            is_fraud=is_fraud,
            fraud_type=fraud_type if is_fraud else None,
        )

        self.transaction_counter += 1
        return transaction

    def _select_merchant_normal(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Select merchant for normal transaction."""
        # 80% chance to use preferred merchant
        if random.random() < 0.8 and user_profile["preferred_merchants"]:
            merchant_id = random.choice(user_profile["preferred_merchants"])
            return next(m for m in self.merchants if m["id"] == merchant_id)
        else:
            return random.choice(self.merchants)

    def _select_merchant_for_fraud(
        self, fraud_type: str, user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Select merchant for fraudulent transaction."""
        pattern = self.fraud_patterns[fraud_type]

        if pattern["merchant_preference"] == "high_risk":
            high_risk_merchants = [m for m in self.merchants if m["risk_score"] > 0.5]
            return random.choice(high_risk_merchants)
        elif pattern["merchant_preference"] == "online_retail":
            online_merchants = [
                m for m in self.merchants if m["category"] == "online_retail"
            ]
            return random.choice(online_merchants)
        else:
            return random.choice(self.merchants)

    def _generate_normal_amount(self, user_profile: Dict[str, Any]) -> float:
        """Generate normal transaction amount."""
        avg_amount = user_profile["avg_transaction_amount"]
        # Log-normal distribution around average
        return max(1.0, random.lognormvariate(avg_amount, 0.5))

    def _generate_fraud_amount(
        self, user_profile: Dict[str, Any], fraud_type: str
    ) -> float:
        """Generate fraudulent transaction amount."""
        pattern = self.fraud_patterns[fraud_type]
        base_amount = user_profile["avg_transaction_amount"]
        multiplier = pattern["amount_multiplier"]

        # Add some randomness
        multiplier *= random.uniform(0.8, 1.5)
        return base_amount * multiplier

    def _generate_normal_location(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate normal transaction location."""
        home = user_profile["home_location"]

        # 90% chance to be within 50km of home
        if random.random() < 0.9:
            lat_offset = random.uniform(-0.5, 0.5)  # ~50km
            lon_offset = random.uniform(-0.5, 0.5)
        else:
            # 10% chance to be further away (business travel, etc.)
            lat_offset = random.uniform(-2, 2)
            lon_offset = random.uniform(-2, 2)

        return {
            "latitude": home["latitude"] + lat_offset,
            "longitude": home["longitude"] + lon_offset,
            "city": home["city"],
            "country": home["country"],
        }

    def _generate_fraud_location(
        self, user_profile: Dict[str, Any], fraud_type: str
    ) -> Dict[str, Any]:
        """Generate fraudulent transaction location."""
        pattern = self.fraud_patterns[fraud_type]
        home = user_profile["home_location"]

        # Generate location far from home
        spread = pattern["geographic_spread"] / 111  # Convert km to degrees
        lat_offset = random.uniform(-spread, spread)
        lon_offset = random.uniform(-spread, spread)

        # Select random city for fraud
        fraud_city = random.choice(self.cities)

        return {
            "latitude": fraud_city["lat"] + random.uniform(-0.1, 0.1),
            "longitude": fraud_city["lon"] + random.uniform(-0.1, 0.1),
            "city": fraud_city["name"],
            "country": fraud_city["country"],
        }

    def _generate_timestamp(self, is_fraud: bool, fraud_type: str) -> datetime:
        """Generate transaction timestamp."""
        if is_fraud and fraud_type:
            pattern = self.fraud_patterns[fraud_type]
            if pattern["time_pattern"] == "random":
                # Random time
                hour = random.randint(0, 23)
                minute = random.randint(0, 59)
            elif pattern["time_pattern"] == "business_hours":
                # Business hours (9 AM - 5 PM)
                hour = random.randint(9, 17)
                minute = random.randint(0, 59)
            else:
                # Normal time pattern
                hour = random.randint(8, 22)  # Most transactions during day
                minute = random.randint(0, 59)
        else:
            # Normal time pattern - more transactions during day
            if random.random() < 0.8:
                hour = random.randint(8, 22)
            else:
                hour = random.randint(0, 7)  # Some night transactions
            minute = random.randint(0, 59)

        # Add some randomness to current time
        timestamp = self.current_time + timedelta(
            hours=random.randint(-1, 1), minutes=random.randint(-30, 30)
        )

        # Set specific hour and minute
        timestamp = timestamp.replace(
            hour=hour, minute=minute, second=random.randint(0, 59)
        )

        return timestamp

    def generate_transactions(
        self, count: int, fraud_rate: float = 0.05
    ) -> List[Transaction]:
        """Generate multiple transactions."""
        transactions = []
        fraud_count = int(count * fraud_rate)

        for i in range(count):
            is_fraud = i < fraud_count
            fraud_type = None

            if is_fraud:
                fraud_type = random.choice(list(self.fraud_patterns.keys()))

            transaction = self.generate_transaction(is_fraud, fraud_type)
            transactions.append(transaction)

        # Shuffle to mix fraud and normal transactions
        random.shuffle(transactions)
        return transactions

    def stream_transactions(
        self, tps: int = 10, duration_hours: int = 1, fraud_rate: float = 0.05
    ) -> Generator[Transaction, None, None]:
        """Stream transactions at specified rate."""
        total_transactions = tps * 3600 * duration_hours
        fraud_count = int(total_transactions * fraud_rate)

        fraud_indices = set(random.sample(range(total_transactions), fraud_count))

        for i in range(total_transactions):
            is_fraud = i in fraud_indices
            fraud_type = None

            if is_fraud:
                fraud_type = random.choice(list(self.fraud_patterns.keys()))

            transaction = self.generate_transaction(is_fraud, fraud_type)

            # Sleep to maintain TPS
            time.sleep(1.0 / tps)

            yield transaction
