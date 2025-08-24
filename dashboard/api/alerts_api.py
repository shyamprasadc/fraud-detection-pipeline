"""
Alerts API endpoints for the fraud detection dashboard.
"""

from flask import Blueprint, jsonify, request
from datetime import datetime
import redis
from typing import Dict, Any, List, Optional
import logging
import json

logger = logging.getLogger(__name__)

alerts_bp = Blueprint("alerts", __name__)


class AlertsAPI:
    """API endpoints for alert management."""

    def __init__(self, redis_client: redis.Redis):
        """Initialize the alerts API."""
        self.redis = redis_client

    def get_alerts(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering."""
        try:
            # Determine which set to query based on status
            if status == "pending":
                key = "alerts:pending"
            elif status == "acknowledged":
                key = "alerts:acknowledged"
            elif status == "resolved":
                key = "alerts:resolved"
            elif priority:
                key = f"alerts:{priority.lower()}"
            else:
                key = "alerts:pending"  # Default to pending alerts

            # Get alert IDs
            alert_ids = self.redis.zrevrange(key, 0, limit - 1)

            # Get alert details
            alerts = []
            for alert_id in alert_ids:
                alert_data = self._get_alert_data(alert_id)
                if alert_data:
                    alerts.append(alert_data)

            return alerts

        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []

    def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific alert by ID."""
        try:
            return self._get_alert_data(alert_id)
        except Exception as e:
            logger.error(f"Error getting alert {alert_id}: {e}")
            return None

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        try:
            alert_key = f"alert:{alert_id}"

            # Update alert status
            self.redis.hset(
                alert_key,
                mapping={
                    "status": "acknowledged",
                    "acknowledged_at": datetime.utcnow().isoformat(),
                    "acknowledged_by": acknowledged_by,
                },
            )

            # Remove from pending alerts
            self.redis.zrem("alerts:pending", alert_id)

            # Add to acknowledged alerts
            acknowledged_key = "alerts:acknowledged"
            self.redis.zadd(acknowledged_key, {alert_id: datetime.utcnow().timestamp()})
            self.redis.expire(acknowledged_key, 86400)  # 24 hours TTL

            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True

        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False

    def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str,
        resolution_notes: Optional[str] = None,
        is_false_positive: bool = False,
    ) -> bool:
        """Resolve an alert."""
        try:
            alert_key = f"alert:{alert_id}"

            # Determine status
            status = "false_positive" if is_false_positive else "resolved"

            # Update alert status
            update_data = {
                "status": status,
                "resolved_at": datetime.utcnow().isoformat(),
                "resolved_by": resolved_by,
            }

            if resolution_notes:
                update_data["resolution_notes"] = resolution_notes

            self.redis.hset(alert_key, mapping=update_data)

            # Remove from all queues
            for queue_key in [
                "alerts:pending",
                "alerts:acknowledged",
                "alerts:high",
                "alerts:urgent",
                "alerts:medium",
                "alerts:low",
            ]:
                self.redis.zrem(queue_key, alert_id)

            # Add to resolved alerts
            resolved_key = "alerts:resolved"
            self.redis.zadd(resolved_key, {alert_id: datetime.utcnow().timestamp()})
            self.redis.expire(resolved_key, 86400)  # 24 hours TTL

            logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            return True

        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        try:
            stats = {
                "total_alerts": 0,
                "pending_alerts": self.redis.zcard("alerts:pending"),
                "acknowledged_alerts": self.redis.zcard("alerts:acknowledged"),
                "resolved_alerts": self.redis.zcard("alerts:resolved"),
                "urgent_alerts": self.redis.zcard("alerts:urgent"),
                "high_alerts": self.redis.zcard("alerts:high"),
                "medium_alerts": self.redis.zcard("alerts:medium"),
                "low_alerts": self.redis.zcard("alerts:low"),
            }

            stats["total_alerts"] = sum(
                [
                    stats["pending_alerts"],
                    stats["acknowledged_alerts"],
                    stats["resolved_alerts"],
                ]
            )

            return stats

        except Exception as e:
            logger.error(f"Error getting alert statistics: {e}")
            return {
                "total_alerts": 0,
                "pending_alerts": 0,
                "acknowledged_alerts": 0,
                "resolved_alerts": 0,
                "urgent_alerts": 0,
                "high_alerts": 0,
                "medium_alerts": 0,
                "low_alerts": 0,
            }

    def _get_alert_data(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Get alert data from Redis."""
        try:
            alert_key = f"alert:{alert_id}"
            alert_data = self.redis.hgetall(alert_key)

            if not alert_data:
                return None

            # Convert string values to appropriate types
            if "fraud_score" in alert_data:
                alert_data["fraud_score"] = float(alert_data["fraud_score"])

            if "created_at" in alert_data:
                alert_data["created_at"] = alert_data["created_at"]

            if "acknowledged_at" in alert_data and alert_data["acknowledged_at"]:
                alert_data["acknowledged_at"] = alert_data["acknowledged_at"]

            if "resolved_at" in alert_data and alert_data["resolved_at"]:
                alert_data["resolved_at"] = alert_data["resolved_at"]

            # Parse JSON fields
            if "rule_triggers" in alert_data:
                try:
                    alert_data["rule_triggers"] = json.loads(
                        alert_data["rule_triggers"]
                    )
                except:
                    alert_data["rule_triggers"] = []

            if "features" in alert_data:
                try:
                    alert_data["features"] = json.loads(alert_data["features"])
                except:
                    alert_data["features"] = {}

            return alert_data

        except Exception as e:
            logger.error(f"Error getting alert data for {alert_id}: {e}")
            return None


# Flask Blueprint routes
@alerts_bp.route("/api/alerts", methods=["GET"])
def get_alerts():
    """Get alerts endpoint."""
    try:
        status = request.args.get("status")
        priority = request.args.get("priority")
        limit = int(request.args.get("limit", 50))

        redis_client = redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        alerts_api = AlertsAPI(redis_client)
        alerts = alerts_api.get_alerts(status=status, priority=priority, limit=limit)

        return jsonify(alerts)
    except Exception as e:
        logger.error(f"Error in get alerts endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@alerts_bp.route("/api/alerts/<alert_id>", methods=["GET"])
def get_alert(alert_id):
    """Get specific alert endpoint."""
    try:
        redis_client = redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        alerts_api = AlertsAPI(redis_client)
        alert = alerts_api.get_alert(alert_id)

        if alert:
            return jsonify(alert)
        else:
            return jsonify({"error": "Alert not found"}), 404
    except Exception as e:
        logger.error(f"Error in get alert endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@alerts_bp.route("/api/alerts/<alert_id>/acknowledge", methods=["POST"])
def acknowledge_alert(alert_id):
    """Acknowledge alert endpoint."""
    try:
        data = request.get_json()
        acknowledged_by = data.get("acknowledged_by", "dashboard_user")

        redis_client = redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        alerts_api = AlertsAPI(redis_client)
        success = alerts_api.acknowledge_alert(alert_id, acknowledged_by)

        if success:
            return jsonify({"status": "success"})
        else:
            return jsonify({"error": "Failed to acknowledge alert"}), 500
    except Exception as e:
        logger.error(f"Error in acknowledge alert endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@alerts_bp.route("/api/alerts/<alert_id>/resolve", methods=["POST"])
def resolve_alert(alert_id):
    """Resolve alert endpoint."""
    try:
        data = request.get_json()
        resolved_by = data.get("resolved_by", "dashboard_user")
        resolution_notes = data.get("resolution_notes")
        is_false_positive = data.get("is_false_positive", False)

        redis_client = redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        alerts_api = AlertsAPI(redis_client)
        success = alerts_api.resolve_alert(
            alert_id, resolved_by, resolution_notes, is_false_positive
        )

        if success:
            return jsonify({"status": "success"})
        else:
            return jsonify({"error": "Failed to resolve alert"}), 500
    except Exception as e:
        logger.error(f"Error in resolve alert endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@alerts_bp.route("/api/alerts/statistics", methods=["GET"])
def get_alert_statistics():
    """Get alert statistics endpoint."""
    try:
        redis_client = redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        alerts_api = AlertsAPI(redis_client)
        stats = alerts_api.get_alert_statistics()

        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error in alert statistics endpoint: {e}")
        return jsonify({"error": str(e)}), 500
