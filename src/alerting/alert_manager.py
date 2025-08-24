"""
Alert manager for fraud detection system.
"""

import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

import redis

from ..models.fraud_score import FraudScore, RiskLevel


class AlertStatus(Enum):
    """Alert status enumeration."""

    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class AlertPriority(Enum):
    """Alert priority enumeration."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Alert:
    """Alert data model."""

    alert_id: str
    transaction_id: str
    user_id: str
    fraud_score: float
    risk_level: RiskLevel
    priority: AlertPriority
    status: AlertStatus = AlertStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    rule_triggers: List[str] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "transaction_id": self.transaction_id,
            "user_id": self.user_id,
            "fraud_score": self.fraud_score,
            "risk_level": self.risk_level.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat()
            if self.acknowledged_at
            else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_by": self.acknowledged_by,
            "resolution_notes": self.resolution_notes,
            "rule_triggers": self.rule_triggers,
            "features": self.features,
        }


class AlertManager:
    """Manages fraud alerts and notifications."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the alert manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Redis configuration
        self.redis_host = config.get("redis_host", "localhost")
        self.redis_port = config.get("redis_port", 6379)
        self.redis_db = config.get("redis_db", 0)
        self.redis_client = None

        # Alert configuration
        self.alert_ttl = config.get("alert_ttl", 86400)  # 24 hours
        self.max_alerts_per_user = config.get("max_alerts_per_user", 10)
        self.alert_deduplication_window = config.get(
            "alert_deduplication_window", 300
        )  # 5 minutes

        # Notification channels
        self.notification_channels = config.get("notification_channels", [])

        # Alert counters
        self.alert_counters = {
            "total_alerts": 0,
            "high_priority_alerts": 0,
            "urgent_alerts": 0,
            "alerts_acknowledged": 0,
            "alerts_resolved": 0,
        }

        # Initialize Redis connection
        self._initialize_redis()

    def _initialize_redis(self):
        """Initialize Redis connection."""
        try:
            import redis

            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=True,
            )
            self.redis_client.ping()
            self.logger.info("Redis connection established")
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    def create_alert(self, fraud_score: FraudScore) -> Optional[Alert]:
        """Create a new alert from fraud score."""
        try:
            # Check if alert should be created
            if not self._should_create_alert(fraud_score):
                return None

            # Generate alert ID
            alert_id = f"alert_{int(time.time())}_{fraud_score.transaction_id[:8]}"

            # Determine priority
            priority = self._determine_priority(fraud_score)

            # Create alert
            alert = Alert(
                alert_id=alert_id,
                transaction_id=fraud_score.transaction_id,
                user_id=fraud_score.user_id,
                fraud_score=fraud_score.fraud_probability,
                risk_level=fraud_score.risk_level,
                priority=priority,
                rule_triggers=fraud_score.rule_triggers,
                features=fraud_score.features,
            )

            # Store alert
            self._store_alert(alert)

            # Update counters
            self.alert_counters["total_alerts"] += 1
            if priority == AlertPriority.HIGH:
                self.alert_counters["high_priority_alerts"] += 1
            elif priority == AlertPriority.URGENT:
                self.alert_counters["urgent_alerts"] += 1

            # Send notifications
            self._send_notifications(alert)

            self.logger.info(
                f"Alert created: {alert_id} for transaction {fraud_score.transaction_id}"
            )
            return alert

        except Exception as e:
            self.logger.error(f"Error creating alert: {e}")
            return None

    def _should_create_alert(self, fraud_score: FraudScore) -> bool:
        """Check if alert should be created based on risk level and deduplication."""
        # Only create alerts for high and critical risk levels
        if fraud_score.risk_level not in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            return False

        # Check deduplication
        if self._is_duplicate_alert(fraud_score):
            return False

        # Check user alert limits
        if self._exceeds_user_alert_limit(fraud_score.user_id):
            return False

        return True

    def _is_duplicate_alert(self, fraud_score: FraudScore) -> bool:
        """Check if this is a duplicate alert."""
        if not self.redis_client:
            return False

        try:
            # Check recent alerts for same user
            user_alerts_key = f"user:{fraud_score.user_id}:recent_alerts"
            recent_alerts = self.redis_client.zrangebyscore(
                user_alerts_key,
                time.time() - self.alert_deduplication_window,
                time.time(),
            )

            # If there are recent alerts, consider it a duplicate
            return len(recent_alerts) > 0

        except Exception as e:
            self.logger.warning(f"Error checking duplicate alerts: {e}")
            return False

    def _exceeds_user_alert_limit(self, user_id: str) -> bool:
        """Check if user has exceeded alert limit."""
        if not self.redis_client:
            return False

        try:
            user_alerts_key = f"user:{user_id}:alerts"
            alert_count = self.redis_client.zcard(user_alerts_key)
            return alert_count >= self.max_alerts_per_user

        except Exception as e:
            self.logger.warning(f"Error checking user alert limit: {e}")
            return False

    def _determine_priority(self, fraud_score: FraudScore) -> AlertPriority:
        """Determine alert priority based on fraud score."""
        if fraud_score.risk_level == RiskLevel.CRITICAL:
            return AlertPriority.URGENT
        elif fraud_score.fraud_probability >= 0.9:
            return AlertPriority.URGENT
        elif fraud_score.fraud_probability >= 0.8:
            return AlertPriority.HIGH
        elif fraud_score.fraud_probability >= 0.6:
            return AlertPriority.MEDIUM
        else:
            return AlertPriority.LOW

    def _store_alert(self, alert: Alert):
        """Store alert in Redis."""
        if not self.redis_client:
            return

        try:
            # Store alert details
            alert_key = f"alert:{alert.alert_id}"
            self.redis_client.hset(alert_key, mapping=alert.to_dict())
            self.redis_client.expire(alert_key, self.alert_ttl)

            # Add to user alerts
            user_alerts_key = f"user:{alert.user_id}:alerts"
            self.redis_client.zadd(
                user_alerts_key, {alert.alert_id: alert.created_at.timestamp()}
            )
            self.redis_client.expire(user_alerts_key, self.alert_ttl)

            # Add to recent alerts for deduplication
            recent_alerts_key = f"user:{alert.user_id}:recent_alerts"
            self.redis_client.zadd(
                recent_alerts_key, {alert.alert_id: alert.created_at.timestamp()}
            )
            self.redis_client.expire(recent_alerts_key, self.alert_deduplication_window)

            # Add to priority queues
            priority_key = f"alerts:{alert.priority.value}"
            self.redis_client.zadd(
                priority_key, {alert.alert_id: alert.created_at.timestamp()}
            )
            self.redis_client.expire(priority_key, self.alert_ttl)

            # Add to pending alerts
            pending_key = "alerts:pending"
            self.redis_client.zadd(
                pending_key, {alert.alert_id: alert.created_at.timestamp()}
            )
            self.redis_client.expire(pending_key, self.alert_ttl)

        except Exception as e:
            self.logger.error(f"Error storing alert: {e}")

    def _send_notifications(self, alert: Alert):
        """Send notifications for the alert."""
        for channel in self.notification_channels:
            try:
                if channel["type"] == "email":
                    self._send_email_notification(alert, channel)
                elif channel["type"] == "slack":
                    self._send_slack_notification(alert, channel)
                elif channel["type"] == "webhook":
                    self._send_webhook_notification(alert, channel)
            except Exception as e:
                self.logger.error(
                    f"Error sending notification via {channel['type']}: {e}"
                )

    def _send_email_notification(self, alert: Alert, channel_config: Dict[str, Any]):
        """Send email notification."""
        # Implementation would use email library (smtplib, etc.)
        self.logger.info(f"Email notification sent for alert {alert.alert_id}")

    def _send_slack_notification(self, alert: Alert, channel_config: Dict[str, Any]):
        """Send Slack notification."""
        # Implementation would use Slack API
        self.logger.info(f"Slack notification sent for alert {alert.alert_id}")

    def _send_webhook_notification(self, alert: Alert, channel_config: Dict[str, Any]):
        """Send webhook notification."""
        # Implementation would use requests library
        self.logger.info(f"Webhook notification sent for alert {alert.alert_id}")

    def get_alerts(
        self,
        status: Optional[AlertStatus] = None,
        priority: Optional[AlertPriority] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Get alerts with optional filtering."""
        if not self.redis_client:
            return []

        try:
            # Determine which set to query
            if status == AlertStatus.PENDING:
                key = "alerts:pending"
            elif priority:
                key = f"alerts:{priority.value}"
            else:
                key = "alerts:pending"  # Default to pending alerts

            # Get alert IDs
            alert_ids = self.redis_client.zrevrange(key, 0, limit - 1)

            # Get alert details
            alerts = []
            for alert_id in alert_ids:
                alert_data = self._get_alert_data(alert_id)
                if alert_data:
                    alerts.append(self._create_alert_from_data(alert_data))

            return alerts

        except Exception as e:
            self.logger.error(f"Error getting alerts: {e}")
            return []

    def _get_alert_data(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Get alert data from Redis."""
        if not self.redis_client:
            return None

        try:
            alert_key = f"alert:{alert_id}"
            alert_data = self.redis_client.hgetall(alert_key)
            return alert_data if alert_data else None
        except Exception as e:
            self.logger.error(f"Error getting alert data: {e}")
            return None

    def _create_alert_from_data(self, alert_data: Dict[str, Any]) -> Alert:
        """Create Alert object from data."""
        return Alert(
            alert_id=alert_data["alert_id"],
            transaction_id=alert_data["transaction_id"],
            user_id=alert_data["user_id"],
            fraud_score=float(alert_data["fraud_score"]),
            risk_level=RiskLevel(alert_data["risk_level"]),
            priority=AlertPriority(alert_data["priority"]),
            status=AlertStatus(alert_data["status"]),
            created_at=datetime.fromisoformat(alert_data["created_at"]),
            acknowledged_at=datetime.fromisoformat(alert_data["acknowledged_at"])
            if alert_data.get("acknowledged_at")
            else None,
            resolved_at=datetime.fromisoformat(alert_data["resolved_at"])
            if alert_data.get("resolved_at")
            else None,
            acknowledged_by=alert_data.get("acknowledged_by"),
            resolution_notes=alert_data.get("resolution_notes"),
            rule_triggers=json.loads(alert_data.get("rule_triggers", "[]")),
            features=json.loads(alert_data.get("features", "{}")),
        )

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        if not self.redis_client:
            return False

        try:
            alert_key = f"alert:{alert_id}"

            # Update alert status
            self.redis_client.hset(
                alert_key,
                mapping={
                    "status": AlertStatus.ACKNOWLEDGED.value,
                    "acknowledged_at": datetime.utcnow().isoformat(),
                    "acknowledged_by": acknowledged_by,
                },
            )

            # Remove from pending alerts
            self.redis_client.zrem("alerts:pending", alert_id)

            # Add to acknowledged alerts
            acknowledged_key = "alerts:acknowledged"
            self.redis_client.zadd(acknowledged_key, {alert_id: time.time()})
            self.redis_client.expire(acknowledged_key, self.alert_ttl)

            self.alert_counters["alerts_acknowledged"] += 1
            self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")

            return True

        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {e}")
            return False

    def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str,
        resolution_notes: Optional[str] = None,
        is_false_positive: bool = False,
    ) -> bool:
        """Resolve an alert."""
        if not self.redis_client:
            return False

        try:
            alert_key = f"alert:{alert_id}"

            # Determine status
            status = (
                AlertStatus.FALSE_POSITIVE
                if is_false_positive
                else AlertStatus.RESOLVED
            )

            # Update alert status
            update_data = {
                "status": status.value,
                "resolved_at": datetime.utcnow().isoformat(),
                "acknowledged_by": resolved_by,
            }

            if resolution_notes:
                update_data["resolution_notes"] = resolution_notes

            self.redis_client.hset(alert_key, mapping=update_data)

            # Remove from all queues
            for queue_key in [
                "alerts:pending",
                "alerts:acknowledged",
                "alerts:high",
                "alerts:urgent",
                "alerts:medium",
                "alerts:low",
            ]:
                self.redis_client.zrem(queue_key, alert_id)

            # Add to resolved alerts
            resolved_key = "alerts:resolved"
            self.redis_client.zadd(resolved_key, {alert_id: time.time()})
            self.redis_client.expire(resolved_key, self.alert_ttl)

            self.alert_counters["alerts_resolved"] += 1
            self.logger.info(f"Alert {alert_id} resolved by {resolved_by}")

            return True

        except Exception as e:
            self.logger.error(f"Error resolving alert: {e}")
            return False

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        if not self.redis_client:
            return self.alert_counters

        try:
            stats = {
                **self.alert_counters,
                "pending_alerts": self.redis_client.zcard("alerts:pending"),
                "acknowledged_alerts": self.redis_client.zcard("alerts:acknowledged"),
                "resolved_alerts": self.redis_client.zcard("alerts:resolved"),
                "urgent_alerts": self.redis_client.zcard("alerts:urgent"),
                "high_alerts": self.redis_client.zcard("alerts:high"),
                "medium_alerts": self.redis_client.zcard("alerts:medium"),
                "low_alerts": self.redis_client.zcard("alerts:low"),
            }

            return stats

        except Exception as e:
            self.logger.error(f"Error getting alert statistics: {e}")
            return self.alert_counters

    def cleanup_old_alerts(self, days: int = 7) -> int:
        """Clean up alerts older than specified days."""
        if not self.redis_client:
            return 0

        try:
            cutoff_time = time.time() - (days * 86400)
            cleaned_count = 0

            # Clean up from all alert sets
            alert_sets = [
                "alerts:pending",
                "alerts:acknowledged",
                "alerts:resolved",
                "alerts:urgent",
                "alerts:high",
                "alerts:medium",
                "alerts:low",
            ]

            for alert_set in alert_sets:
                old_alerts = self.redis_client.zrangebyscore(alert_set, 0, cutoff_time)
                if old_alerts:
                    self.redis_client.zremrangebyscore(alert_set, 0, cutoff_time)
                    cleaned_count += len(old_alerts)

            self.logger.info(f"Cleaned up {cleaned_count} old alerts")
            return cleaned_count

        except Exception as e:
            self.logger.error(f"Error cleaning up old alerts: {e}")
            return 0
