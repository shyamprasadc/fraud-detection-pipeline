# Fraud Detection Pipeline

A real-time fraud detection system that ingests payment transactions, applies ML-based fraud scoring, and triggers alerts for suspicious activities.

## Features

- **Real-time Transaction Processing**: Handle 100+ transactions per second with <200ms latency
- **ML-Based Fraud Detection**: Random Forest model with >90% accuracy
- **Feature Engineering**: Real-time velocity, geographic, and behavioral features
- **Alert System**: Configurable risk thresholds with email/Slack notifications
- **Monitoring Dashboard**: Real-time metrics and transaction explorer
- **Data Simulator**: Generate realistic transaction data for testing

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Simulator│───▶│  Kafka Producer │───▶│  Kafka Topic    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Alert Manager  │◀───│ Fraud Detector  │◀───│Transaction Proc.│
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dashboard     │    │   Feature Store │    │   ML Model      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Kafka
- Redis

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd fraud-detection-pipeline
```

2. Set up environment:

```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start the infrastructure:

```bash
docker-compose up -d
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Train the model:

```bash
python scripts/train_model.py
```

6. Start the pipeline:

```bash
python scripts/start_pipeline.py
```

7. Access the dashboard:

```
http://localhost:5000
```

## Project Structure

```
fraud-detection-pipeline/
├── README.md                 # This file
├── docker-compose.yml        # Infrastructure setup
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore rules
├── data/                    # Sample data and configurations
├── src/                     # Core application code
├── dashboard/               # Web dashboard
├── config/                  # Configuration files
├── scripts/                 # Utility scripts
└── notebooks/               # Jupyter notebooks
```

## Configuration

### Environment Variables

- `KAFKA_BOOTSTRAP_SERVERS`: Kafka broker addresses
- `REDIS_HOST`: Redis server host
- `REDIS_PORT`: Redis server port
- `MODEL_PATH`: Path to trained ML model
- `ALERT_EMAIL`: Email for fraud alerts
- `SLACK_WEBHOOK`: Slack webhook URL

### Risk Thresholds

- **HIGH**: ≥80% fraud probability
- **MEDIUM**: 50-79% fraud probability
- **LOW**: 30-49% fraud probability

## API Endpoints

### Dashboard API

- `GET /api/metrics` - Real-time metrics
- `GET /api/transactions` - Transaction list
- `GET /api/alerts` - Alert list
- `POST /api/transactions/{id}/review` - Review transaction

## Monitoring

### Key Metrics

- Transactions per second
- Fraud detection rate
- False positive rate
- Model inference latency
- Alert volume

### Alerts

- High-risk transaction notifications
- Model performance degradation
- System health monitoring

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Quality

```bash
flake8 src/
black src/
```

### Model Retraining

```bash
python scripts/train_model.py --retrain
```

## Performance

- **Throughput**: 100+ TPS
- **Latency**: <200ms end-to-end
- **Accuracy**: >90% fraud detection
- **Availability**: 99.9% uptime

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
