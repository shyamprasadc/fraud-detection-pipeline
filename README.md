# Fraud Detection Pipeline

A real-time fraud detection system that ingests payment transactions, applies ML-based fraud scoring, and triggers alerts for suspicious activities.

## Features

- **Real-time Transaction Processing**: Handle 100+ transactions per second with <200ms latency
- **ML-Based Fraud Detection**: Random Forest and XGBoost models with >90% accuracy
- **Feature Engineering**: Real-time velocity, geographic, temporal, and behavioral features
- **Alert System**: Configurable risk thresholds with email/Slack/Discord notifications
- **Monitoring Dashboard**: Real-time metrics, interactive charts, and transaction explorer
- **Data Simulator**: Generate realistic transaction data with fraud patterns
- **Configuration Management**: YAML-based configuration with environment variable overrides
- **Jupyter Notebooks**: Data exploration and model training notebooks
- **Demo Script**: Complete pipeline demonstration with metrics

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
- Kafka (provided via Docker)
- Redis (provided via Docker)

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

### Alternative: Automated Setup

Use the setup script for a complete automated installation:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Demo Mode

Run a complete demonstration of the pipeline:

```bash
python scripts/demo.py --duration 10 --tps 20
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
│   └── feature_definitions.json
├── src/                     # Core application code
│   ├── models/              # Data models
│   ├── ingestion/           # Data ingestion and simulation
│   ├── processing/          # Feature engineering and ML pipeline
│   ├── ml/                  # Model training and serving
│   └── alerting/            # Alert management
├── dashboard/               # Web dashboard
│   ├── app.py               # Flask application
│   ├── api/                 # API endpoints
│   ├── templates/           # HTML templates
│   └── static/              # CSS/JS assets
├── config/                  # Configuration files
│   ├── pipeline_config.yaml # Main configuration
│   └── config_loader.py     # Configuration management
├── scripts/                 # Utility scripts
│   ├── setup.sh             # Automated setup
│   ├── train_model.py       # Model training
│   ├── start_pipeline.py    # Pipeline startup
│   └── demo.py              # Demo showcase
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
└── models/                  # Trained model storage
```

## Configuration

### Environment Variables

- `KAFKA_BOOTSTRAP_SERVERS`: Kafka broker addresses
- `REDIS_HOST`: Redis server host
- `REDIS_PORT`: Redis server port
- `MODEL_PATH`: Path to trained ML model
- `ALERT_EMAIL`: Email for fraud alerts
- `SLACK_WEBHOOK_URL`: Slack webhook URL
- `DISCORD_WEBHOOK_URL`: Discord webhook URL
- `RISK_THRESHOLD_HIGH/MEDIUM/LOW`: Risk thresholds
- `DASHBOARD_HOST/PORT`: Dashboard configuration
- `SIMULATOR_TPS/FRAUD_RATE`: Data simulator settings

### Risk Thresholds

- **HIGH**: ≥80% fraud probability
- **MEDIUM**: 50-79% fraud probability
- **LOW**: 30-49% fraud probability

## API Endpoints

### Dashboard API

- `GET /api/metrics` - Real-time system metrics
- `GET /api/transactions` - Transaction list with filtering
- `GET /api/transactions/{id}` - Specific transaction details
- `GET /api/alerts` - Alert list with filtering
- `POST /api/alerts/{id}/acknowledge` - Acknowledge alert
- `POST /api/alerts/{id}/resolve` - Resolve alert
- `GET /api/health` - System health check
- `GET /api/statistics` - System statistics

## Monitoring

### Key Metrics

- Transactions per second
- Fraud detection rate
- False positive rate
- Model inference latency
- Alert volume
- System health status

### Alerts

- High-risk transaction notifications
- Configurable risk thresholds
- Email, Slack, and Discord notifications
- Alert lifecycle management (acknowledge, resolve)

## Development

### Jupyter Notebooks

Explore data and train models interactively:

```bash
cd notebooks
jupyter notebook
```

Available notebooks:

- `01_data_exploration.ipynb` - Data analysis and visualization
- `02_model_training.ipynb` - Model training and evaluation

### Model Training

Train models with various options:

```bash
# Basic training
python scripts/train_model.py

# With hyperparameter tuning
python scripts/train_model.py --tune-hyperparameters

# Custom parameters
python scripts/train_model.py --transactions 20000 --fraud-rate 0.03 --model-type xgboost
```

### Configuration Management

The system uses YAML-based configuration with environment variable overrides:

```python
from config.config_loader import load_config

config = load_config()
kafka_config = config.get_kafka_config()
redis_config = config.get_redis_config()
```

## Performance

- **Throughput**: 100+ TPS
- **Latency**: <200ms end-to-end
- **Accuracy**: >90% fraud detection
- **Availability**: 99.9% uptime target

## Infrastructure

### Docker Services

- **Kafka**: Message streaming platform
- **Redis**: Feature store and caching
- **Kafka-UI**: Web interface for Kafka management
- **Redis Commander**: Web interface for Redis management

### Monitoring Tools

- **Kafka-UI**: Available at http://localhost:8080
- **Redis Commander**: Available at http://localhost:8081
- **Dashboard**: Available at http://localhost:5000

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests (when test framework is implemented)
5. Submit a pull request

## Future Enhancements

### Planned Features

- **Testing Framework**: Unit and integration tests
- **Code Quality Tools**: Automated linting and formatting
- **Transaction Review API**: Manual review workflow
- **Model Retraining Pipeline**: Automated model updates
- **Advanced Analytics**: More detailed fraud pattern analysis
- **Multi-tenancy**: Support for multiple organizations
- **API Documentation**: OpenAPI/Swagger documentation

### Performance Improvements

- **Horizontal Scaling**: Multi-instance deployment
- **Caching Optimization**: Enhanced Redis caching strategies
- **Stream Processing**: Apache Flink integration
- **Real-time Analytics**: Advanced streaming analytics

## License

MIT License - see LICENSE file for details.
