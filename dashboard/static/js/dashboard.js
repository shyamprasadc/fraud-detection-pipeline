// Fraud Detection Dashboard JavaScript

class FraudDetectionDashboard {
  constructor() {
    this.updateInterval = 5000; // 5 seconds
    this.charts = {};
    this.isInitialized = false;

    this.init();
  }

  init() {
    console.log("Initializing Fraud Detection Dashboard...");

    // Initialize charts
    this.initCharts();

    // Load initial data
    this.loadMetrics();
    this.loadAlerts();
    this.loadTransactions();

    // Set up periodic updates
    setInterval(() => {
      this.loadMetrics();
    }, this.updateInterval);

    setInterval(() => {
      this.loadAlerts();
    }, this.updateInterval * 2);

    setInterval(() => {
      this.loadTransactions();
    }, this.updateInterval * 3);

    this.isInitialized = true;
  }

  async loadMetrics() {
    try {
      const response = await fetch("/api/metrics");
      const metrics = await response.json();

      this.updateMetricsDisplay(metrics);
      this.updateSystemStatus(metrics);
    } catch (error) {
      console.error("Error loading metrics:", error);
      this.showError("Failed to load metrics");
    }
  }

  async loadAlerts() {
    try {
      const response = await fetch("/api/alerts?limit=10");
      const alerts = await response.json();

      this.updateAlertsTable(alerts);
      this.updateAlertCharts(alerts);
    } catch (error) {
      console.error("Error loading alerts:", error);
    }
  }

  async loadTransactions() {
    try {
      const response = await fetch("/api/transactions?limit=20");
      const transactions = await response.json();

      this.updateTransactionsTable(transactions);
    } catch (error) {
      console.error("Error loading transactions:", error);
    }
  }

  updateMetricsDisplay(metrics) {
    // Update metric cards
    const transactionVolume = document.getElementById("transaction-volume");
    const fraudRate = document.getElementById("fraud-rate");
    const pendingAlerts = document.getElementById("pending-alerts");
    const systemHealth = document.getElementById("system-health");

    if (transactionVolume && metrics.transactions) {
      transactionVolume.textContent =
        metrics.transactions.processing_rate?.toFixed(2) || "0.00";
    }

    if (fraudRate && metrics.alerts) {
      const totalAlerts = metrics.alerts.total_alerts || 0;
      const totalTransactions = metrics.transactions?.recent_transactions || 1;
      const rate = ((totalAlerts / totalTransactions) * 100).toFixed(2);
      fraudRate.textContent = rate;
    }

    if (pendingAlerts && metrics.alerts) {
      pendingAlerts.textContent = metrics.alerts.pending_alerts || 0;
    }

    if (systemHealth && metrics.system) {
      systemHealth.textContent = metrics.system.status || "Unknown";
    }
  }

  updateSystemStatus(metrics) {
    const statusElement = document.getElementById("system-status");
    if (!statusElement) return;

    const status = metrics.system?.status || "unknown";
    const healthStatus = metrics.system?.components || {};

    // Remove existing classes
    statusElement.className = "status-badge";

    // Add appropriate class based on status
    if (
      status === "healthy" &&
      Object.values(healthStatus).every((s) => s === "healthy")
    ) {
      statusElement.classList.add("healthy");
      statusElement.textContent = "Healthy";
    } else if (
      status === "healthy" ||
      Object.values(healthStatus).some((s) => s === "healthy")
    ) {
      statusElement.classList.add("warning");
      statusElement.textContent = "Warning";
    } else {
      statusElement.classList.add("error");
      statusElement.textContent = "Error";
    }
  }

  updateAlertsTable(alerts) {
    const tbody = document.getElementById("alerts-table-body");
    if (!tbody) return;

    if (alerts.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7">No alerts found</td></tr>';
      return;
    }

    tbody.innerHTML = alerts
      .map(
        (alert) => `
            <tr>
                <td>${alert.alert_id}</td>
                <td>${alert.transaction_id}</td>
                <td>${alert.user_id}</td>
                <td><span class="risk-badge ${alert.risk_level.toLowerCase()}">${
          alert.risk_level
        }</span></td>
                <td>${(alert.fraud_score * 100).toFixed(1)}%</td>
                <td>${new Date(alert.created_at).toLocaleString()}</td>
                <td>
                    ${
                      alert.status === "pending"
                        ? `
                        <button class="action-btn acknowledge" onclick="dashboard.acknowledgeAlert('${alert.alert_id}')">
                            Acknowledge
                        </button>
                        <button class="action-btn resolve" onclick="dashboard.resolveAlert('${alert.alert_id}')">
                            Resolve
                        </button>
                        <button class="action-btn false-positive" onclick="dashboard.markFalsePositive('${alert.alert_id}')">
                            False Positive
                        </button>
                    `
                        : alert.status
                    }
                </td>
            </tr>
        `
      )
      .join("");
  }

  updateTransactionsTable(transactions) {
    const tbody = document.getElementById("transactions-table-body");
    if (!tbody) return;

    if (transactions.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6">No transactions found</td></tr>';
      return;
    }

    tbody.innerHTML = transactions
      .map(
        (txn) => `
            <tr>
                <td>${txn.transaction_id || "N/A"}</td>
                <td>${txn.user_id || "N/A"}</td>
                <td>$${parseFloat(txn.amount || 0).toFixed(2)}</td>
                <td>${txn.merchant_name || txn.merchant_id || "N/A"}</td>
                <td>${
                  txn.fraud_score
                    ? (txn.fraud_score * 100).toFixed(1) + "%"
                    : "N/A"
                }</td>
                <td>
                    ${
                      txn.fraud_score > 0.8
                        ? '<span class="risk-badge high">High Risk</span>'
                        : txn.fraud_score > 0.5
                        ? '<span class="risk-badge medium">Medium Risk</span>'
                        : '<span class="risk-badge low">Low Risk</span>'
                    }
                </td>
            </tr>
        `
      )
      .join("");
  }

  initCharts() {
    // Initialize fraud distribution chart
    const fraudCtx = document.getElementById("fraud-distribution-chart");
    if (fraudCtx) {
      this.charts.fraudDistribution = new Chart(fraudCtx, {
        type: "doughnut",
        data: {
          labels: ["Low Risk", "Medium Risk", "High Risk", "Critical"],
          datasets: [
            {
              data: [0, 0, 0, 0],
              backgroundColor: ["#10b981", "#f59e0b", "#ef4444", "#dc2626"],
              borderWidth: 0,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "bottom",
            },
          },
        },
      });
    }

    // Initialize alert priority chart
    const alertCtx = document.getElementById("alert-priority-chart");
    if (alertCtx) {
      this.charts.alertPriority = new Chart(alertCtx, {
        type: "bar",
        data: {
          labels: ["Low", "Medium", "High", "Urgent"],
          datasets: [
            {
              label: "Alerts",
              data: [0, 0, 0, 0],
              backgroundColor: ["#10b981", "#f59e0b", "#ef4444", "#dc2626"],
              borderWidth: 0,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
            },
          },
          plugins: {
            legend: {
              display: false,
            },
          },
        },
      });
    }
  }

  updateAlertCharts(alerts) {
    // Update fraud distribution chart
    if (this.charts.fraudDistribution) {
      const distribution = {
        low: 0,
        medium: 0,
        high: 0,
        critical: 0,
      };

      alerts.forEach((alert) => {
        const score = alert.fraud_score;
        if (score < 0.3) distribution.low++;
        else if (score < 0.5) distribution.medium++;
        else if (score < 0.8) distribution.high++;
        else distribution.critical++;
      });

      this.charts.fraudDistribution.data.datasets[0].data = [
        distribution.low,
        distribution.medium,
        distribution.high,
        distribution.critical,
      ];
      this.charts.fraudDistribution.update();
    }

    // Update alert priority chart
    if (this.charts.alertPriority) {
      const priorities = {
        low: 0,
        medium: 0,
        high: 0,
        urgent: 0,
      };

      alerts.forEach((alert) => {
        const priority = alert.priority.toLowerCase();
        if (priorities.hasOwnProperty(priority)) {
          priorities[priority]++;
        }
      });

      this.charts.alertPriority.data.datasets[0].data = [
        priorities.low,
        priorities.medium,
        priorities.high,
        priorities.urgent,
      ];
      this.charts.alertPriority.update();
    }
  }

  async acknowledgeAlert(alertId) {
    try {
      const response = await fetch(`/api/alerts/${alertId}/acknowledge`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          acknowledged_by: "dashboard_user",
        }),
      });

      if (response.ok) {
        this.showSuccess("Alert acknowledged successfully");
        this.loadAlerts(); // Refresh alerts
      } else {
        this.showError("Failed to acknowledge alert");
      }
    } catch (error) {
      console.error("Error acknowledging alert:", error);
      this.showError("Failed to acknowledge alert");
    }
  }

  async resolveAlert(alertId) {
    try {
      const response = await fetch(`/api/alerts/${alertId}/resolve`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          resolved_by: "dashboard_user",
          resolution_notes: "Resolved via dashboard",
        }),
      });

      if (response.ok) {
        this.showSuccess("Alert resolved successfully");
        this.loadAlerts(); // Refresh alerts
      } else {
        this.showError("Failed to resolve alert");
      }
    } catch (error) {
      console.error("Error resolving alert:", error);
      this.showError("Failed to resolve alert");
    }
  }

  async markFalsePositive(alertId) {
    try {
      const response = await fetch(`/api/alerts/${alertId}/resolve`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          resolved_by: "dashboard_user",
          is_false_positive: true,
          resolution_notes: "Marked as false positive",
        }),
      });

      if (response.ok) {
        this.showSuccess("Alert marked as false positive");
        this.loadAlerts(); // Refresh alerts
      } else {
        this.showError("Failed to mark alert as false positive");
      }
    } catch (error) {
      console.error("Error marking false positive:", error);
      this.showError("Failed to mark alert as false positive");
    }
  }

  showSuccess(message) {
    this.showNotification(message, "success");
  }

  showError(message) {
    this.showNotification(message, "error");
  }

  showNotification(message, type = "info") {
    // Create notification element
    const notification = document.createElement("div");
    notification.className = `notification ${type}`;
    notification.textContent = message;
    notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 20px;
            border-radius: 6px;
            color: white;
            font-weight: 500;
            z-index: 1000;
            animation: slideIn 0.3s ease;
            background-color: ${
              type === "success"
                ? "#10b981"
                : type === "error"
                ? "#ef4444"
                : "#3b82f6"
            };
        `;

    // Add to page
    document.body.appendChild(notification);

    // Remove after 3 seconds
    setTimeout(() => {
      notification.style.animation = "slideOut 0.3s ease";
      setTimeout(() => {
        if (notification.parentNode) {
          notification.parentNode.removeChild(notification);
        }
      }, 300);
    }, 3000);
  }
}

// Add CSS animations
const style = document.createElement("style");
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);

// Initialize dashboard when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  window.dashboard = new FraudDetectionDashboard();
});
