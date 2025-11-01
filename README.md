# Agentic AI Inventory Management System

A sophisticated multi-agent system for intelligent inventory management using AI pattern analysis and automation.

## Features

- **Multi-Agent Architecture**: Specialized agents for planning, execution, and reflection
- **Pattern Analysis**: AI-powered demand forecasting and trend analysis
- **Automated Reordering**: Smart inventory replenishment based on predictive models
- **Real-time Monitoring**: Continuous tracking of inventory levels and performance
- **Interactive Dashboard**: Streamlit-based UI for visualization and control

## System Architecture

```
agentic-inventory/
├── .env                          # Environment variables (secure)
├── .git/                         # Git version control
├── .gitignore                    # Git ignore rules
├── agents/                       # AI agent modules
│   ├── __init__.py              # Package initialization
│   ├── executor.py              # ExecutorAgent for order execution
│   ├── planner.py               # PlannerAgent for inventory planning
│   ├── reflector.py             # ReflectorAgent for system optimization
│   └── __pycache__/             # Python cache
├── config.json                   # System configuration settings
├── data/                         # Data storage
│   ├── sales.csv                # Historical sales transactions
│   └── stock.csv                # Current inventory levels
├── models/                       # ML models and business logic
│   ├── __init__.py              # Package initialization
│   ├── predictor.py             # Enhanced ML demand prediction
│   ├── notification_system.py   # Email alert system
│   ├── shop_operations.py       # Business operations logic
│   ├── demand_predictor_*.pkl   # Trained ML models (4 files)
│   ├── last_training.txt        # ML training metadata
│   └── __pycache__/             # Python cache
├── ui/                          # User interface
│   └── app.py                   # Professional Streamlit dashboard
├── inventory_system.log         # System logging
├── main.py                      # Main application entry point
├── QUICK_START.md               # Quick setup guide
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── run.bat                      # Windows launcher script
├── run.sh                       # Unix/Linux launcher script
├── STATUS.md                    # System status documentation
└── venv/                        # Python virtual environment
```

## Installation

1. Clone or download the project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables** (⚠️ IMPORTANT for security):
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env with your credentials (NEVER commit this file!)
   # See SECURITY.md for detailed setup instructions
   ```
4. Run the application:
   ```bash
   python main.py
   ```

> **🔒 Security Note:** Never commit your `.env` file! It contains sensitive credentials. See [SECURITY.md](SECURITY.md) for detailed security guidelines.

## Usage

### Running the System
```bash
python main.py
```

### Starting the Dashboard
```bash
streamlit run ui/app.py
```

## Agents Overview

### 1. Planner Agent
- Analyzes historical data and market trends
- Creates strategic inventory plans
- Optimizes reorder points and quantities

### 2. Executor Agent
- Implements planned actions
- Places orders with suppliers
- Updates inventory records

### 3. Reflector Agent
- Evaluates system performance
- Identifies improvement opportunities
- Provides feedback for optimization

## Key Components

- **Demand Forecasting**: Uses machine learning to predict future demand
- **Inventory Optimization**: Minimizes costs while preventing stockouts
- **Supplier Management**: Evaluates and selects optimal suppliers
- **Automated Reporting**: Generates insights and performance metrics

## License

MIT License
