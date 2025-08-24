# Jupyter Notebooks

This directory contains Jupyter notebooks for data analysis, model training, and pipeline exploration.

## Available Notebooks

### 1. Data Exploration (`01_data_exploration.ipynb`)

**Purpose**: Explore and understand the transaction data and fraud patterns.

**Contents**:

- Data generation and loading
- Transaction amount analysis
- Temporal pattern analysis
- Merchant category analysis
- Geographic analysis
- Feature engineering preview
- Key insights and summary

**Usage**:

```bash
jupyter notebook 01_data_exploration.ipynb
```

### 2. Model Training (`02_model_training.ipynb`)

**Purpose**: Train and evaluate fraud detection models.

**Contents**:

- Data preparation and feature engineering
- Model training (Random Forest, XGBoost)
- Model evaluation and comparison
- Feature importance analysis
- Model validation
- Model persistence and metadata

**Usage**:

```bash
jupyter notebook 02_model_training.ipynb
```

## Prerequisites

Before running the notebooks, ensure you have:

1. **Dependencies installed**:

   ```bash
   pip install -r ../requirements.txt
   ```

2. **Infrastructure running**:

   ```bash
   docker-compose up -d
   ```

3. **Redis connection**:

   - Redis should be running on localhost:6379
   - The notebooks will connect to Redis for feature storage

4. **Jupyter installed**:
   ```bash
   pip install jupyter
   ```

## Setup Instructions

1. **Start the infrastructure**:

   ```bash
   cd ..
   docker-compose up -d
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Start Jupyter**:

   ```bash
   cd notebooks
   jupyter notebook
   ```

4. **Run notebooks in order**:
   - Start with `01_data_exploration.ipynb`
   - Then run `02_model_training.ipynb`

## Notebook Features

### Data Exploration Notebook

- **Interactive visualizations** using matplotlib and seaborn
- **Statistical analysis** of transaction patterns
- **Feature correlation analysis**
- **Fraud pattern identification**
- **Geographic and temporal analysis**

### Model Training Notebook

- **Multiple model comparison** (Random Forest vs XGBoost)
- **Hyperparameter tuning** capabilities
- **Cross-validation** and performance metrics
- **Feature importance analysis**
- **Model persistence** and metadata storage

## Output Files

The notebooks generate:

- **Trained models** in the `../models/` directory
- **Visualizations** and charts
- **Performance metrics** and analysis reports
- **Feature importance** rankings

## Customization

You can customize the notebooks by:

1. **Adjusting data generation parameters**:

   - Number of transactions
   - Fraud rate
   - Time period
   - User and merchant counts

2. **Modifying model parameters**:

   - Model types
   - Hyperparameters
   - Feature sets
   - Evaluation metrics

3. **Adding new analysis**:
   - Additional visualizations
   - Custom metrics
   - New feature engineering
   - Model comparison

## Troubleshooting

### Common Issues

1. **Redis Connection Error**:

   - Ensure Redis is running: `docker-compose ps`
   - Check Redis connection: `redis-cli ping`

2. **Import Errors**:

   - Ensure you're in the notebooks directory
   - Check that `../src` is in the Python path
   - Verify all dependencies are installed

3. **Memory Issues**:
   - Reduce the number of transactions generated
   - Use smaller sample sizes for analysis
   - Clear variables between notebook sections

### Getting Help

- Check the main project README for setup instructions
- Review the logs in the `../logs/` directory
- Ensure all infrastructure services are running

## Next Steps

After running the notebooks:

1. **Deploy models** to production using the scripts
2. **Set up monitoring** for model performance
3. **Configure alerts** for fraud detection
4. **Implement real-time scoring** pipeline
5. **Set up automated retraining** workflows
