"""
Transactions API endpoints for the fraud detection dashboard.
"""

from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
import redis
from typing import Dict, Any, List, Optional
import logging
import json

logger = logging.getLogger(__name__)

transactions_bp = Blueprint("transactions", __name__)


class TransactionsAPI:
    """API endpoints for transaction management."""

    def __init__(self, redis_client: redis.Redis):
        """Initialize the transactions API."""
        self.redis = redis_client

    def get_transactions(
        self,
        limit: int = 100,
        user_id: Optional[str] = None,
        merchant_id: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent transactions with optional filtering."""
        try:
            # Get recent transaction IDs
            transaction_ids = self.redis.zrevrange("recent_transactions", 0, limit - 1)

            transactions = []
            for txn_id in transaction_ids:
                transaction_data = self._get_transaction_data(txn_id)
                if transaction_data:
                    # Apply filters
                    if user_id and transaction_data.get("user_id") != user_id:
                        continue
                    if (
                        merchant_id
                        and transaction_data.get("merchant_id") != merchant_id
                    ):
                        continue
                    if risk_level and not self._matches_risk_level(
                        transaction_data, risk_level
                    ):
                        continue

                    transactions.append(transaction_data)

            return transactions

        except Exception as e:
            logger.error(f"Error getting transactions: {e}")
            return []

    def get_transaction(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific transaction by ID."""
        try:
            transaction_data = self._get_transaction_data(transaction_id)
            if not transaction_data:
                return None

            # Get associated fraud score
            fraud_score_data = self._get_fraud_score_data(transaction_id)
            if fraud_score_data:
                transaction_data["fraud_score"] = fraud_score_data

            # Get associated alert
            alert_data = self._get_alert_data(transaction_id)
            if alert_data:
                transaction_data["alert"] = alert_data

            return transaction_data

        except Exception as e:
            logger.error(f"Error getting transaction {transaction_id}: {e}")
            return None

    def get_transaction_statistics(self) -> Dict[str, Any]:
        """Get transaction statistics."""
        try:
            # Get recent transaction count
            recent_transactions = self.redis.zcard("recent_transactions")

            # Get transaction rate (last hour)
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            hourly_transactions = self.redis.zcount(
                "recent_transactions",
                one_hour_ago.timestamp(),
                datetime.utcnow().timestamp(),
            )

            # Get transaction rate (last 24 hours)
            one_day_ago = datetime.utcnow() - timedelta(days=1)
            daily_transactions = self.redis.zcount(
                "recent_transactions",
                one_day_ago.timestamp(),
                datetime.utcnow().timestamp(),
            )

            # Get fraud score distribution
            fraud_distribution = self._get_fraud_score_distribution()

            # Get amount statistics
            amount_stats = self._get_amount_statistics()

            return {
                "recent_transactions": recent_transactions,
                "hourly_transactions": hourly_transactions,
                "daily_transactions": daily_transactions,
                "hourly_rate": hourly_transactions / 3600.0,
                "daily_rate": daily_transactions / 86400.0,
                "fraud_distribution": fraud_distribution,
                "amount_statistics": amount_stats,
            }

        except Exception as e:
            logger.error(f"Error getting transaction statistics: {e}")
            return {
                "recent_transactions": 0,
                "hourly_transactions": 0,
                "daily_transactions": 0,
                "hourly_rate": 0.0,
                "daily_rate": 0.0,
                "fraud_distribution": {},
                "amount_statistics": {},
            }

    def search_transactions(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search transactions by various criteria."""
        try:
            # This is a simplified search - in a real implementation,
            # you might use Elasticsearch or similar
            all_transactions = self.get_transactions(limit=1000)

            matching_transactions = []
            query_lower = query.lower()

            for transaction in all_transactions:
                # Search in various fields
                searchable_fields = [
                    str(transaction.get("user_id", "")),
                    str(transaction.get("merchant_id", "")),
                    str(transaction.get("merchant_name", "")),
                    str(transaction.get("transaction_id", "")),
                    str(transaction.get("card_last_four", "")),
                ]

                if any(query_lower in field.lower() for field in searchable_fields):
                    matching_transactions.append(transaction)

                if len(matching_transactions) >= limit:
                    break

            return matching_transactions

        except Exception as e:
            logger.error(f"Error searching transactions: {e}")
            return []

    def _get_transaction_data(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get transaction data from Redis."""
        try:
            transaction_key = f"transaction:{transaction_id}"
            transaction_data = self.redis.hgetall(transaction_key)

            if not transaction_data:
                return None

            # Convert numeric fields
            if "amount" in transaction_data:
                transaction_data["amount"] = float(transaction_data["amount"])

            # Parse location data
            if "location" in transaction_data:
                try:
                    transaction_data["location"] = json.loads(
                        transaction_data["location"]
                    )
                except:
                    transaction_data["location"] = None

            # Parse timestamp
            if "timestamp" in transaction_data:
                transaction_data["timestamp"] = transaction_data["timestamp"]

            return transaction_data

        except Exception as e:
            logger.error(f"Error getting transaction data for {transaction_id}: {e}")
            return None

    def _get_fraud_score_data(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get fraud score data for a transaction."""
        try:
            fraud_score_key = f"fraud_score:{transaction_id}"
            fraud_score_data = self.redis.hgetall(fraud_score_key)

            if not fraud_score_data:
                return None

            # Convert numeric fields
            if "fraud_probability" in fraud_score_data:
                fraud_score_data["fraud_probability"] = float(
                    fraud_score_data["fraud_probability"]
                )

            # Parse features
            if "features" in fraud_score_data:
                try:
                    fraud_score_data["features"] = json.loads(
                        fraud_score_data["features"]
                    )
                except:
                    fraud_score_data["features"] = {}

            # Parse rule triggers
            if "rule_triggers" in fraud_score_data:
                try:
                    fraud_score_data["rule_triggers"] = json.loads(
                        fraud_score_data["rule_triggers"]
                    )
                except:
                    fraud_score_data["rule_triggers"] = []

            return fraud_score_data

        except Exception as e:
            logger.error(f"Error getting fraud score data for {transaction_id}: {e}")
            return None

    def _get_alert_data(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get alert data for a transaction."""
        try:
            alert_id = self.redis.get(f"transaction_alert:{transaction_id}")
            if not alert_id:
                return None

            alert_key = f"alert:{alert_id}"
            alert_data = self.redis.hgetall(alert_key)

            if not alert_data:
                return None

            # Convert numeric fields
            if "fraud_score" in alert_data:
                alert_data["fraud_score"] = float(alert_data["fraud_score"])

            return alert_data

        except Exception as e:
            logger.error(f"Error getting alert data for {transaction_id}: {e}")
            return None

    def _matches_risk_level(
        self, transaction_data: Dict[str, Any], risk_level: str
    ) -> bool:
        """Check if transaction matches the specified risk level."""
        try:
            fraud_score_data = self._get_fraud_score_data(
                transaction_data.get("transaction_id", "")
            )
            if not fraud_score_data:
                return False

            fraud_probability = fraud_score_data.get("fraud_probability", 0.0)

            if risk_level == "low":
                return fraud_probability < 0.3
            elif risk_level == "medium":
                return 0.3 <= fraud_probability < 0.5
            elif risk_level == "high":
                return 0.5 <= fraud_probability < 0.8
            elif risk_level == "critical":
                return fraud_probability >= 0.8
            else:
                return False

        except Exception:
            return False

    def _get_fraud_score_distribution(self) -> Dict[str, int]:
        """Get distribution of fraud scores."""
        try:
            recent_scores = self.redis.zrange(
                "recent_fraud_scores", 0, -1, withscores=True
            )

            distribution = {"low": 0, "medium": 0, "high": 0, "critical": 0}

            for score_data in recent_scores:
                score = float(score_data[1])
                if score < 0.3:
                    distribution["low"] += 1
                elif score < 0.5:
                    distribution["medium"] += 1
                elif score < 0.8:
                    distribution["high"] += 1
                else:
                    distribution["critical"] += 1

            return distribution

        except Exception as e:
            logger.warning(f"Error getting fraud score distribution: {e}")
            return {"low": 0, "medium": 0, "high": 0, "critical": 0}

    def _get_amount_statistics(self) -> Dict[str, Any]:
        """Get transaction amount statistics."""
        try:
            # Get recent transactions
            recent_transactions = self.get_transactions(limit=1000)

            if not recent_transactions:
                return {
                    "total_amount": 0.0,
                    "average_amount": 0.0,
                    "min_amount": 0.0,
                    "max_amount": 0.0,
                    "transaction_count": 0,
                }

            amounts = [
                txn.get("amount", 0.0)
                for txn in recent_transactions
                if txn.get("amount")
            ]

            if not amounts:
                return {
                    "total_amount": 0.0,
                    "average_amount": 0.0,
                    "min_amount": 0.0,
                    "max_amount": 0.0,
                    "transaction_count": 0,
                }

            return {
                "total_amount": sum(amounts),
                "average_amount": sum(amounts) / len(amounts),
                "min_amount": min(amounts),
                "max_amount": max(amounts),
                "transaction_count": len(amounts),
            }

        except Exception as e:
            logger.warning(f"Error getting amount statistics: {e}")
            return {
                "total_amount": 0.0,
                "average_amount": 0.0,
                "min_amount": 0.0,
                "max_amount": 0.0,
                "transaction_count": 0,
            }


# Flask Blueprint routes
@transactions_bp.route("/api/transactions", methods=["GET"])
def get_transactions():
    """Get transactions endpoint."""
    try:
        limit = int(request.args.get("limit", 100))
        user_id = request.args.get("user_id")
        merchant_id = request.args.get("merchant_id")
        risk_level = request.args.get("risk_level")

        redis_client = redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        transactions_api = TransactionsAPI(redis_client)
        transactions = transactions_api.get_transactions(
            limit=limit, user_id=user_id, merchant_id=merchant_id, risk_level=risk_level
        )

        return jsonify(transactions)
    except Exception as e:
        logger.error(f"Error in get transactions endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@transactions_bp.route("/api/transactions/<transaction_id>", methods=["GET"])
def get_transaction(transaction_id):
    """Get specific transaction endpoint."""
    try:
        redis_client = redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        transactions_api = TransactionsAPI(redis_client)
        transaction = transactions_api.get_transaction(transaction_id)

        if transaction:
            return jsonify(transaction)
        else:
            return jsonify({"error": "Transaction not found"}), 404
    except Exception as e:
        logger.error(f"Error in get transaction endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@transactions_bp.route("/api/transactions/statistics", methods=["GET"])
def get_transaction_statistics():
    """Get transaction statistics endpoint."""
    try:
        redis_client = redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        transactions_api = TransactionsAPI(redis_client)
        stats = transactions_api.get_transaction_statistics()

        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error in transaction statistics endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@transactions_bp.route("/api/transactions/search", methods=["GET"])
def search_transactions():
    """Search transactions endpoint."""
    try:
        query = request.args.get("q", "")
        limit = int(request.args.get("limit", 50))

        if not query:
            return jsonify([])

        redis_client = redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        transactions_api = TransactionsAPI(redis_client)
        transactions = transactions_api.search_transactions(query, limit=limit)

        return jsonify(transactions)
    except Exception as e:
        logger.error(f"Error in search transactions endpoint: {e}")
        return jsonify({"error": str(e)}), 500
