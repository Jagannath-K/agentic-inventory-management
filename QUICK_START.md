# Quick Start Guide

## Getting Started

1. **First Run - Interactive Mode**
   ```
   python main.py
   ```

2. **Launch Dashboard**
   ```
   python main.py dashboard
   ```
   or
   ```
   streamlit run ui/app.py
   ```

3. **Run Single Inventory Cycle**
   ```
   python main.py cycle
   ```

4. **Start Automated Scheduler**
   ```
   python main.py scheduler
   ```

## Dashboard Features

- **Overview**: System metrics and recent activity
- **AI Planning**: Generate and review inventory plans
- **Execution**: Execute orders automatically
- **Analytics**: Performance analysis and insights
- **Forecasting**: Demand prediction and trends

## Configuration

Edit `config.json` to customize:
- Automation levels
- Budget limits
- Notification settings
- Forecasting parameters

## Data Files

- `data/sales.csv`: Historical sales data
- `data/stock.csv`: Current inventory levels

## Troubleshooting

1. **Import Errors**: Run `pip install -r requirements.txt`
2. **Data Errors**: Verify CSV file formats and headers
3. **Permission Errors**: Check file and directory permissions
4. **Dashboard Issues**: Try `streamlit run ui/app.py` directly
