"""
Flask dashboard for fraud detection pipeline monitoring.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_cors import CORS
import redis

# Import our modules
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.alerting.alert_manager import AlertManager, AlertStatus, AlertPriority
from src.processing.transaction_processor import TransactionProcessor
from src.models.fraud_score import RiskLevel


class FraudDetectionDashboard:
    """Main dashboard application."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the dashboard."""
        self.config = config
        self.app = Flask(__name__)
        CORS(self.app)

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=config.get("redis_host", "localhost"),
            port=config.get("redis_port", 6379),
            db=config.get("redis_db", 0),
            decode_responses=True,
        )

        # Initialize components
        self.alert_manager = AlertManager(config.get("alert_config", {}))

        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Register Flask routes."""

        @self.app.route("/")
        def index():
            """Main dashboard page."""
            return render_template("fraud_dashboard.html")

        @self.app.route("/api/metrics")
        def get_metrics():
            """Get real-time metrics."""
            try:
                metrics = self._get_system_metrics()
                return jsonify(metrics)
            except Exception as e:
                self.logger.error(f"Error getting metrics: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/alerts")
        def get_alerts():
            """Get alerts with optional filtering."""
            try:
                status = request.args.get("status")
                priority = request.args.get("priority")
                limit = int(request.args.get("limit", 50))

                alerts = self.alert_manager.get_alerts(
                    status=AlertStatus(status) if status else None,
                    priority=AlertPriority(priority) if priority else None,
                    limit=limit,
                )

                return jsonify([alert.to_dict() for alert in alerts])
            except Exception as e:
                self.logger.error(f"Error getting alerts: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/alerts/<alert_id>/acknowledge", methods=["POST"])
        def acknowledge_alert(alert_id):
            """Acknowledge an alert."""
            try:
                data = request.get_json()
                acknowledged_by = data.get("acknowledged_by", "dashboard_user")

                success = self.alert_manager.acknowledge_alert(
                    alert_id, acknowledged_by
                )

                if success:
                    return jsonify({"status": "success"})
                else:
                    return jsonify({"error": "Failed to acknowledge alert"}), 400
            except Exception as e:
                self.logger.error(f"Error acknowledging alert: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/alerts/<alert_id>/resolve", methods=["POST"])
        def resolve_alert(alert_id):
            """Resolve an alert."""
            try:
                data = request.get_json()
                resolved_by = data.get("resolved_by", "dashboard_user")
                resolution_notes = data.get("resolution_notes")
                is_false_positive = data.get("is_false_positive", False)

                success = self.alert_manager.resolve_alert(
                    alert_id, resolved_by, resolution_notes, is_false_positive
                )

                if success:
                    return jsonify({"status": "success"})
                else:
                    return jsonify({"error": "Failed to resolve alert"}), 400
            except Exception as e:
                self.logger.error(f"Error resolving alert: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/transactions")
        def get_transactions():
            """Get recent transactions."""
            try:
                limit = int(request.args.get("limit", 100))
                transactions = self._get_recent_transactions(limit)
                return jsonify(transactions)
            except Exception as e:
                self.logger.error(f"Error getting transactions: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/transactions/<transaction_id>")
        def get_transaction(transaction_id):
            """Get specific transaction details."""
            try:
                transaction = self._get_transaction_details(transaction_id)
                if transaction:
                    return jsonify(transaction)
                else:
                    return jsonify({"error": "Transaction not found"}), 404
            except Exception as e:
                self.logger.error(f"Error getting transaction: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/health")
        def health_check():
            """Health check endpoint."""
            try:
                health = self._get_health_status()
                return jsonify(health)
            except Exception as e:
                self.logger.error(f"Error in health check: {e}")
                return jsonify({"status": "unhealthy", "error": str(e)}), 500

        @self.app.route("/api/statistics")
        def get_statistics():
            """Get system statistics."""
            try:
                stats = self._get_system_statistics()
                return jsonify(stats)
            except Exception as e:
                self.logger.error(f"Error getting statistics: {e}")
                return jsonify({"error": str(e)}), 500

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get real-time system metrics."""
        try:
            # Get alert statistics
            alert_stats = self.alert_manager.get_alert_statistics()

            # Get Redis metrics
            redis_info = self.redis_client.info()

            # Get transaction metrics (simplified)
            transaction_metrics = self._get_transaction_metrics()

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "alerts": {
                    "total_alerts": alert_stats.get("total_alerts", 0),
                    "pending_alerts": alert_stats.get("pending_alerts", 0),
                    "urgent_alerts": alert_stats.get("urgent_alerts", 0),
                    "high_alerts": alert_stats.get("high_alerts", 0),
                    "medium_alerts": alert_stats.get("medium_alerts", 0),
                    "low_alerts": alert_stats.get("low_alerts", 0),
                },
                "transactions": transaction_metrics,
                "redis": {
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "used_memory_human": redis_info.get("used_memory_human", "0B"),
                    "total_commands_processed": redis_info.get(
                        "total_commands_processed", 0
                    ),
                },
                "system": {"uptime": self._get_uptime(), "status": "healthy"},
            }
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "status": "error",
            }

    def _get_transaction_metrics(self) -> Dict[str, Any]:
        """Get transaction processing metrics."""
        try:
            # Get recent transaction count from Redis
            recent_transactions = self.redis_client.zcard("recent_transactions")

            # Get fraud score distribution
            fraud_scores = self._get_fraud_score_distribution()

            return {
                "recent_transactions": recent_transactions,
                "fraud_score_distribution": fraud_scores,
                "processing_rate": self._estimate_processing_rate(),
            }
        except Exception as e:
            self.logger.warning(f"Error getting transaction metrics: {e}")
            return {
                "recent_transactions": 0,
                "fraud_score_distribution": {},
                "processing_rate": 0,
            }

    def _get_fraud_score_distribution(self) -> Dict[str, int]:
        """Get distribution of fraud scores."""
        try:
            # Get recent fraud scores from Redis
            recent_scores = self.redis_client.zrange(
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
            self.logger.warning(f"Error getting fraud score distribution: {e}")
            return {"low": 0, "medium": 0, "high": 0, "critical": 0}

    def _estimate_processing_rate(self) -> float:
        """Estimate current processing rate (transactions per second)."""
        try:
            # Get transaction count from last hour
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            recent_transactions = self.redis_client.zcount(
                "recent_transactions",
                one_hour_ago.timestamp(),
                datetime.utcnow().timestamp(),
            )

            # Calculate TPS
            tps = recent_transactions / 3600.0
            return round(tps, 2)
        except Exception as e:
            self.logger.warning(f"Error estimating processing rate: {e}")
            return 0.0

    def _get_recent_transactions(self, limit: int) -> List[Dict[str, Any]]:
        """Get recent transactions from Redis."""
        try:
            # Get recent transaction IDs
            transaction_ids = self.redis_client.zrevrange(
                "recent_transactions", 0, limit - 1
            )

            transactions = []
            for txn_id in transaction_ids:
                txn_data = self.redis_client.hgetall(f"transaction:{txn_id}")
                if txn_data:
                    transactions.append(txn_data)

            return transactions
        except Exception as e:
            self.logger.error(f"Error getting recent transactions: {e}")
            return []

    def _get_transaction_details(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed transaction information."""
        try:
            # Get transaction data
            txn_data = self.redis_client.hgetall(f"transaction:{transaction_id}")
            if not txn_data:
                return None

            # Get fraud score
            fraud_score_data = self.redis_client.hgetall(
                f"fraud_score:{transaction_id}"
            )

            # Get alert if exists
            alert_data = None
            alert_id = self.redis_client.get(f"transaction_alert:{transaction_id}")
            if alert_id:
                alert_data = self.redis_client.hgetall(f"alert:{alert_id}")

            return {
                "transaction": txn_data,
                "fraud_score": fraud_score_data,
                "alert": alert_data,
            }
        except Exception as e:
            self.logger.error(f"Error getting transaction details: {e}")
            return None

    def _get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        try:
            # Test Redis connection
            redis_healthy = self.redis_client.ping()

            # Check alert manager
            alert_manager_healthy = self.alert_manager.redis_client is not None

            return {
                "status": "healthy"
                if redis_healthy and alert_manager_healthy
                else "unhealthy",
                "components": {
                    "redis": "healthy" if redis_healthy else "unhealthy",
                    "alert_manager": "healthy"
                    if alert_manager_healthy
                    else "unhealthy",
                },
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    def _get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            # Get alert statistics
            alert_stats = self.alert_manager.get_alert_statistics()

            # Get Redis statistics
            redis_info = self.redis_client.info()

            # Get time-based statistics
            now = datetime.utcnow()
            hour_ago = now - timedelta(hours=1)
            day_ago = now - timedelta(days=1)

            hourly_alerts = self.redis_client.zcount(
                "alerts:pending", hour_ago.timestamp(), now.timestamp()
            )

            daily_alerts = self.redis_client.zcount(
                "alerts:pending", day_ago.timestamp(), now.timestamp()
            )

            return {
                "alerts": {
                    **alert_stats,
                    "hourly_alerts": hourly_alerts,
                    "daily_alerts": daily_alerts,
                },
                "redis": {
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "used_memory_human": redis_info.get("used_memory_human", "0B"),
                    "total_commands_processed": redis_info.get(
                        "total_commands_processed", 0
                    ),
                    "keyspace_hits": redis_info.get("keyspace_hits", 0),
                    "keyspace_misses": redis_info.get("keyspace_misses", 0),
                },
                "system": {
                    "uptime": self._get_uptime(),
                    "last_updated": now.isoformat(),
                },
            }
        except Exception as e:
            self.logger.error(f"Error getting system statistics: {e}")
            return {"error": str(e)}

    def _get_uptime(self) -> str:
        """Get system uptime (simplified)."""
        try:
            # Get start time from Redis
            start_time = self.redis_client.get("system:start_time")
            if start_time:
                start_dt = datetime.fromisoformat(start_time)
                uptime = datetime.utcnow() - start_dt
                return str(uptime).split(".")[0]  # Remove microseconds
            else:
                return "Unknown"
        except Exception as e:
            self.logger.warning(f"Error getting uptime: {e}")
            return "Unknown"

    def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
        """Run the Flask application."""
        self.logger.info(f"Starting dashboard on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def create_dashboard_app(config: Dict[str, Any]) -> Flask:
    """Create and configure the dashboard Flask app."""
    dashboard = FraudDetectionDashboard(config)
    return dashboard.app


if __name__ == "__main__":
    # Default configuration
    config = {
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_db": 0,
        "alert_config": {"redis_host": "localhost", "redis_port": 6379, "redis_db": 0},
    }

    # Create and run dashboard
    dashboard = FraudDetectionDashboard(config)
    dashboard.run(debug=True)
