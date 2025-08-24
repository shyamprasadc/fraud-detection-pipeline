"""
Feature engineering for fraud detection.
"""

import math
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict

import redis
import numpy as np

from ..models.transaction import Transaction, Location


class FeatureEngine:
    """Real-time feature engineering for fraud detection."""

    def __init__(self, redis_client: redis.Redis, config: Dict[str, Any] = None):
        """Initialize the feature engine."""
        self.redis = redis_client
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Feature definitions
        self.features = [
            # Velocity features
            "txn_count_1h",
            "txn_count_24h",
            "amount_sum_1h",
            "amount_sum_24h",
            # Amount features
            "amount_zscore",
            "amount_percentile",
            "amount_deviation",
            # Geographic features
            "distance_from_home",
            "new_merchant",
            "new_location",
            # Temporal features
            "hour_of_day",
            "is_weekend",
            "is_business_hours",
            # Merchant features
            "merchant_risk_score",
            "merchant_category_risk",
            # Behavioral features
            "user_avg_amount",
            "user_transaction_frequency",
            "user_preferred_merchants_match",
        ]

        # Cache TTL settings
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # 1 hour

        # Distance calculation cache
        self.distance_cache = {}

    def calculate_features(self, transaction: Transaction) -> Dict[str, float]:
        """Calculate all features for a transaction."""
        try:
            features = {}

            # Get user profile from cache
            user_profile = self._get_user_profile(transaction.user_id)

            # Velocity features
            features.update(self._calculate_velocity_features(transaction))

            # Amount features
            features.update(self._calculate_amount_features(transaction, user_profile))

            # Geographic features
            features.update(
                self._calculate_geographic_features(transaction, user_profile)
            )

            # Temporal features
            features.update(self._calculate_temporal_features(transaction))

            # Merchant features
            features.update(self._calculate_merchant_features(transaction))

            # Behavioral features
            features.update(
                self._calculate_behavioral_features(transaction, user_profile)
            )

            # Update user profile with new transaction
            self._update_user_profile(transaction, features)

            return features

        except Exception as e:
            self.logger.error(
                f"Error calculating features for transaction {transaction.transaction_id}: {e}"
            )
            # Return default features
            return self._get_default_features()

    def _calculate_velocity_features(
        self, transaction: Transaction
    ) -> Dict[str, float]:
        """Calculate velocity-based features."""
        features = {}

        try:
            # Get transaction history from cache
            user_key = f"user:{transaction.user_id}"

            # 1-hour window
            txn_1h = self._get_transaction_count(user_key, "1h", transaction.timestamp)
            amount_1h = self._get_amount_sum(user_key, "1h", transaction.timestamp)

            # 24-hour window
            txn_24h = self._get_transaction_count(
                user_key, "24h", transaction.timestamp
            )
            amount_24h = self._get_amount_sum(user_key, "24h", transaction.timestamp)

            features.update(
                {
                    "txn_count_1h": float(txn_1h),
                    "txn_count_24h": float(txn_24h),
                    "amount_sum_1h": float(amount_1h),
                    "amount_sum_24h": float(amount_24h),
                }
            )

        except Exception as e:
            self.logger.warning(f"Error calculating velocity features: {e}")
            features.update(
                {
                    "txn_count_1h": 0.0,
                    "txn_count_24h": 0.0,
                    "amount_sum_1h": 0.0,
                    "amount_sum_24h": 0.0,
                }
            )

        return features

    def _calculate_amount_features(
        self, transaction: Transaction, user_profile: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate amount-based features."""
        features = {}

        try:
            user_avg_amount = user_profile.get("avg_transaction_amount", 100.0)
            user_amount_std = user_profile.get("amount_std", 50.0)

            # Z-score
            if user_amount_std > 0:
                zscore = (transaction.amount - user_avg_amount) / user_amount_std
            else:
                zscore = 0.0

            # Percentile (simplified)
            percentile = min(
                1.0,
                max(
                    0.0, (transaction.amount - user_avg_amount) / (user_avg_amount * 3)
                ),
            )

            # Deviation from average
            deviation = abs(transaction.amount - user_avg_amount) / max(
                user_avg_amount, 1.0
            )

            features.update(
                {
                    "amount_zscore": zscore,
                    "amount_percentile": percentile,
                    "amount_deviation": deviation,
                }
            )

        except Exception as e:
            self.logger.warning(f"Error calculating amount features: {e}")
            features.update(
                {
                    "amount_zscore": 0.0,
                    "amount_percentile": 0.5,
                    "amount_deviation": 0.0,
                }
            )

        return features

    def _calculate_geographic_features(
        self, transaction: Transaction, user_profile: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate geographic features."""
        features = {}

        try:
            # Distance from home
            home_location = user_profile.get("home_location")
            if home_location and transaction.location:
                distance = self._calculate_distance(
                    home_location["latitude"],
                    home_location["longitude"],
                    transaction.location.latitude,
                    transaction.location.longitude,
                )
            else:
                distance = 0.0

            # New merchant check
            user_merchants_key = f"user:{transaction.user_id}:merchants"
            new_merchant = not self.redis.sismember(
                user_merchants_key, transaction.merchant_id
            )

            # New location check
            user_locations_key = f"user:{transaction.user_id}:locations"
            location_hash = f"{transaction.location.latitude:.2f},{transaction.location.longitude:.2f}"
            new_location = not self.redis.sismember(user_locations_key, location_hash)

            features.update(
                {
                    "distance_from_home": distance,
                    "new_merchant": float(new_merchant),
                    "new_location": float(new_location),
                }
            )

        except Exception as e:
            self.logger.warning(f"Error calculating geographic features: {e}")
            features.update(
                {"distance_from_home": 0.0, "new_merchant": 0.0, "new_location": 0.0}
            )

        return features

    def _calculate_temporal_features(
        self, transaction: Transaction
    ) -> Dict[str, float]:
        """Calculate temporal features."""
        features = {}

        try:
            # Hour of day
            hour = transaction.timestamp.hour

            # Weekend check
            is_weekend = transaction.timestamp.weekday() >= 5

            # Business hours (9 AM - 5 PM)
            is_business_hours = 9 <= hour <= 17

            features.update(
                {
                    "hour_of_day": float(hour),
                    "is_weekend": float(is_weekend),
                    "is_business_hours": float(is_business_hours),
                }
            )

        except Exception as e:
            self.logger.warning(f"Error calculating temporal features: {e}")
            features.update(
                {"hour_of_day": 12.0, "is_weekend": 0.0, "is_business_hours": 1.0}
            )

        return features

    def _calculate_merchant_features(
        self, transaction: Transaction
    ) -> Dict[str, float]:
        """Calculate merchant-based features."""
        features = {}

        try:
            # Get merchant risk score from cache
            merchant_key = f"merchant:{transaction.merchant_id}"
            merchant_risk = self.redis.hget(merchant_key, "risk_score")
            merchant_risk = float(merchant_risk) if merchant_risk else 0.1

            # Category risk (simplified)
            category_risk_map = {
                "jewelry": 0.3,
                "electronics": 0.2,
                "travel": 0.25,
                "online_retail": 0.15,
                "gas_station": 0.12,
                "food_beverage": 0.05,
                "healthcare": 0.06,
                "entertainment": 0.05,
                "gaming": 0.1,
                "transportation": 0.15,
                "retail": 0.08,
            }
            category_risk = category_risk_map.get(transaction.merchant_category, 0.1)

            features.update(
                {
                    "merchant_risk_score": merchant_risk,
                    "merchant_category_risk": category_risk,
                }
            )

        except Exception as e:
            self.logger.warning(f"Error calculating merchant features: {e}")
            features.update({"merchant_risk_score": 0.1, "merchant_category_risk": 0.1})

        return features

    def _calculate_behavioral_features(
        self, transaction: Transaction, user_profile: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate behavioral features."""
        features = {}

        try:
            # User average amount
            user_avg_amount = user_profile.get("avg_transaction_amount", 100.0)

            # Transaction frequency (transactions per day)
            user_frequency = user_profile.get("daily_transaction_count", 2.0)

            # Preferred merchants match
            preferred_merchants = user_profile.get("preferred_merchants", [])
            preferred_match = float(transaction.merchant_id in preferred_merchants)

            features.update(
                {
                    "user_avg_amount": user_avg_amount,
                    "user_transaction_frequency": user_frequency,
                    "user_preferred_merchants_match": preferred_match,
                }
            )

        except Exception as e:
            self.logger.warning(f"Error calculating behavioral features: {e}")
            features.update(
                {
                    "user_avg_amount": 100.0,
                    "user_transaction_frequency": 2.0,
                    "user_preferred_merchants_match": 0.0,
                }
            )

        return features

    def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile from cache."""
        try:
            user_key = f"user:{user_id}:profile"
            profile_data = self.redis.hgetall(user_key)

            if profile_data:
                return {
                    "avg_transaction_amount": float(
                        profile_data.get(b"avg_amount", 100.0)
                    ),
                    "amount_std": float(profile_data.get(b"amount_std", 50.0)),
                    "daily_transaction_count": float(
                        profile_data.get(b"daily_txn_count", 2.0)
                    ),
                    "home_location": self._parse_location(
                        profile_data.get(b"home_location", b"{}")
                    ),
                    "preferred_merchants": self._parse_list(
                        profile_data.get(b"preferred_merchants", b"[]")
                    ),
                }
            else:
                # Return default profile
                return {
                    "avg_transaction_amount": 100.0,
                    "amount_std": 50.0,
                    "daily_transaction_count": 2.0,
                    "home_location": None,
                    "preferred_merchants": [],
                }

        except Exception as e:
            self.logger.warning(f"Error getting user profile for {user_id}: {e}")
            return {
                "avg_transaction_amount": 100.0,
                "amount_std": 50.0,
                "daily_transaction_count": 2.0,
                "home_location": None,
                "preferred_merchants": [],
            }

    def _update_user_profile(
        self, transaction: Transaction, features: Dict[str, float]
    ):
        """Update user profile with new transaction data."""
        try:
            user_key = f"user:{transaction.user_id}:profile"

            # Update transaction history
            self._add_transaction_to_history(transaction)

            # Update merchant history
            merchant_key = f"user:{transaction.user_id}:merchants"
            self.redis.sadd(merchant_key, transaction.merchant_id)
            self.redis.expire(merchant_key, self.cache_ttl)

            # Update location history
            if transaction.location:
                location_key = f"user:{transaction.user_id}:locations"
                location_hash = f"{transaction.location.latitude:.2f},{transaction.location.longitude:.2f}"
                self.redis.sadd(location_key, location_hash)
                self.redis.expire(location_key, self.cache_ttl)

            # Update average amount (simplified moving average)
            current_avg = self.redis.hget(user_key, "avg_amount")
            if current_avg:
                current_avg = float(current_avg)
                new_avg = (current_avg * 0.9) + (transaction.amount * 0.1)
            else:
                new_avg = transaction.amount

            self.redis.hset(user_key, "avg_amount", new_avg)
            self.redis.expire(user_key, self.cache_ttl)

        except Exception as e:
            self.logger.warning(f"Error updating user profile: {e}")

    def _add_transaction_to_history(self, transaction: Transaction):
        """Add transaction to user's transaction history."""
        try:
            user_key = f"user:{transaction.user_id}"

            # Add to 1-hour window
            txn_1h_key = f"{user_key}:transactions:1h"
            self.redis.zadd(
                txn_1h_key,
                {transaction.transaction_id: transaction.timestamp.timestamp()},
            )
            self.redis.expire(txn_1h_key, 3600)  # 1 hour TTL

            # Add to 24-hour window
            txn_24h_key = f"{user_key}:transactions:24h"
            self.redis.zadd(
                txn_24h_key,
                {transaction.transaction_id: transaction.timestamp.timestamp()},
            )
            self.redis.expire(txn_24h_key, 86400)  # 24 hours TTL

            # Add amount to sum
            amount_1h_key = f"{user_key}:amounts:1h"
            self.redis.zadd(
                amount_1h_key,
                {transaction.transaction_id: transaction.timestamp.timestamp()},
            )
            self.redis.expire(amount_1h_key, 3600)

            amount_24h_key = f"{user_key}:amounts:24h"
            self.redis.zadd(
                amount_24h_key,
                {transaction.transaction_id: transaction.timestamp.timestamp()},
            )
            self.redis.expire(amount_24h_key, 86400)

        except Exception as e:
            self.logger.warning(f"Error adding transaction to history: {e}")

    def _get_transaction_count(
        self, user_key: str, window: str, timestamp: datetime
    ) -> int:
        """Get transaction count for time window."""
        try:
            if window == "1h":
                key = f"{user_key}:transactions:1h"
                cutoff = timestamp.timestamp() - 3600
            elif window == "24h":
                key = f"{user_key}:transactions:24h"
                cutoff = timestamp.timestamp() - 86400
            else:
                return 0

            # Remove old transactions
            self.redis.zremrangebyscore(key, 0, cutoff)

            # Count remaining transactions
            return self.redis.zcard(key)

        except Exception as e:
            self.logger.warning(f"Error getting transaction count: {e}")
            return 0

    def _get_amount_sum(self, user_key: str, window: str, timestamp: datetime) -> float:
        """Get amount sum for time window."""
        try:
            if window == "1h":
                key = f"{user_key}:amounts:1h"
                cutoff = timestamp.timestamp() - 3600
            elif window == "24h":
                key = f"{user_key}:amounts:24h"
                cutoff = timestamp.timestamp() - 86400
            else:
                return 0.0

            # Remove old transactions
            self.redis.zremrangebyscore(key, 0, cutoff)

            # Get remaining transactions (simplified - in real implementation, store amounts)
            return float(self.redis.zcard(key)) * 100.0  # Approximate

        except Exception as e:
            self.logger.warning(f"Error getting amount sum: {e}")
            return 0.0

    def _calculate_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points using Haversine formula."""
        cache_key = f"{lat1:.2f},{lon1:.2f},{lat2:.2f},{lon2:.2f}"

        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]

        try:
            # Haversine formula
            R = 6371  # Earth's radius in kilometers

            lat1_rad = math.radians(lat1)
            lon1_rad = math.radians(lon1)
            lat2_rad = math.radians(lat2)
            lon2_rad = math.radians(lon2)

            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad

            a = (
                math.sin(dlat / 2) ** 2
                + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
            )
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = R * c

            # Cache result
            self.distance_cache[cache_key] = distance

            # Limit cache size
            if len(self.distance_cache) > 1000:
                self.distance_cache.clear()

            return distance

        except Exception as e:
            self.logger.warning(f"Error calculating distance: {e}")
            return 0.0

    def _parse_location(self, location_str: bytes) -> Optional[Dict[str, Any]]:
        """Parse location string from Redis."""
        try:
            import json

            return json.loads(location_str.decode("utf-8"))
        except:
            return None

    def _parse_list(self, list_str: bytes) -> List[str]:
        """Parse list string from Redis."""
        try:
            import json

            return json.loads(list_str.decode("utf-8"))
        except:
            return []

    def _get_default_features(self) -> Dict[str, float]:
        """Get default feature values."""
        return {feature: 0.0 for feature in self.features}
