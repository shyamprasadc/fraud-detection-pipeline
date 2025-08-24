#!/bin/bash

# Fraud Detection Pipeline Setup Script
# This script sets up the environment for the fraud detection pipeline

set -e

echo "🚀 Setting up Fraud Detection Pipeline..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION is installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating project directories..."
    
    mkdir -p logs
    mkdir -p models
    mkdir -p data
    mkdir -p config
    
    print_success "Directories created"
}

# Setup Python virtual environment
setup_virtual_env() {
    print_status "Setting up Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    print_success "Virtual environment setup complete"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install requirements
    pip install -r requirements.txt
    
    print_success "Dependencies installed"
}

# Setup environment file
setup_env_file() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f "env.example" ]; then
            cp env.example .env
            print_success "Environment file created from template"
            print_warning "Please edit .env file with your configuration"
        else
            print_warning "No env.example found, creating basic .env file"
            cat > .env << EOF
# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_PAYMENT_TRANSACTIONS=payment-transactions
KAFKA_TOPIC_FRAUD_ALERTS=fraud-alerts
KAFKA_CONSUMER_GROUP_FRAUD_DETECTION=fraud-detection-group
KAFKA_CONSUMER_GROUP_ANALYTICS=analytics-group

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# ML Model Configuration
MODEL_PATH=models/fraud_detection_model.pkl
MODEL_VERSION=v1.0
FEATURE_STORE_PATH=data/feature_definitions.json

# Alert Configuration
ALERT_EMAIL=fraud-alerts@company.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR/DISCORD/WEBHOOK

# Risk Thresholds
RISK_THRESHOLD_HIGH=0.8
RISK_THRESHOLD_MEDIUM=0.5
RISK_THRESHOLD_LOW=0.3

# Dashboard Configuration
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=5000
DASHBOARD_DEBUG=False

# Data Simulator Configuration
SIMULATOR_TPS=50
SIMULATOR_FRAUD_RATE=0.05
SIMULATOR_DURATION_HOURS=24

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json

# Performance Configuration
MAX_CONCURRENT_TRANSACTIONS=1000
FEATURE_CACHE_TTL=3600
MODEL_INFERENCE_TIMEOUT=100
EOF
        fi
    else
        print_warning "Environment file already exists"
    fi
}

# Start infrastructure services
start_infrastructure() {
    print_status "Starting infrastructure services (Kafka, Redis)..."
    
    # Start services using docker-compose
    docker-compose up -d
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 30
    
    # Check if services are running
    if docker-compose ps | grep -q "Up"; then
        print_success "Infrastructure services started"
    else
        print_error "Failed to start infrastructure services"
        exit 1
    fi
}

# Train initial model
train_model() {
    print_status "Training initial fraud detection model..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Train model
    python scripts/train_model.py --transactions 5000 --fraud-rate 0.05
    
    print_success "Initial model trained"
}

# Setup complete
setup_complete() {
    echo ""
    echo "🎉 Fraud Detection Pipeline Setup Complete!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Edit .env file with your configuration"
    echo "2. Start the pipeline: python scripts/start_pipeline.py --start-producer"
    echo "3. Access the dashboard: http://localhost:5000"
    echo "4. Monitor Kafka UI: http://localhost:8080"
    echo "5. Monitor Redis Commander: http://localhost:8081"
    echo ""
    echo "📚 Documentation:"
    echo "- README.md for detailed instructions"
    echo "- Check logs/ directory for application logs"
    echo ""
    echo "🛠️  Useful Commands:"
    echo "- Train model: python scripts/train_model.py"
    echo "- Start pipeline: python scripts/start_pipeline.py"
    echo "- Start dashboard: python dashboard/app.py"
    echo "- Stop services: docker-compose down"
    echo ""
}

# Main setup function
main() {
    echo "🔧 Fraud Detection Pipeline Setup"
    echo "=================================="
    echo ""
    
    # Check prerequisites
    check_docker
    check_python
    
    # Setup environment
    create_directories
    setup_virtual_env
    install_dependencies
    setup_env_file
    
    # Start infrastructure
    start_infrastructure
    
    # Train model (optional)
    read -p "Do you want to train an initial model? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        train_model
    else
        print_warning "Skipping model training. Run 'python scripts/train_model.py' later."
    fi
    
    # Setup complete
    setup_complete
}

# Run main function
main "$@"
