"""
Setup script for Agentic Inventory Management System
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(e.stderr)
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "models", 
        "logs",
        "agents",
        "ui"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def install_requirements():
    """Install Python requirements"""
    requirements = [
        "pandas==2.1.4",
        "numpy==1.24.3", 
        "scikit-learn==1.3.2",
        "matplotlib==3.8.2",
        "seaborn==0.13.0",
        "streamlit==1.29.0",
        "plotly==5.17.0",
        "python-dotenv==1.0.0",
        "aiofiles==23.2.1",
        "joblib"
    ]
    
    print("\n📦 Installing Python packages...")
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install each requirement
    for package in requirements:
        if not run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}"):
            print(f"⚠️  Warning: Failed to install {package}")
    
    print("✅ Package installation completed")

def setup_environment():
    """Setup environment file"""
    env_content = """# Environment variables for Agentic Inventory System
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic-inventory

# Database settings (if using external DB)
DATABASE_URL=sqlite:///inventory.db

# Email settings for notifications
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

# Alert thresholds
LOW_STOCK_THRESHOLD=10
CRITICAL_STOCK_THRESHOLD=5
REORDER_BUFFER_DAYS=7
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file")
    else:
        print("✅ .env file already exists")

def verify_installation():
    """Verify that all components are properly installed"""
    print("\n🔍 Verifying installation...")
    
    # Check if data files exist
    data_files = ['data/sales.csv', 'data/stock.csv', 'data/suppliers.csv']
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
    
    # Try importing key modules
    try:
        import pandas
        import numpy
        import sklearn
        import streamlit
        import plotly
        print("✅ All required packages can be imported")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def create_quick_start_guide():
    """Create a quick start guide"""
    guide_content = """# Quick Start Guide

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
- `data/suppliers.csv`: Supplier information

## Getting Help

- Check the logs in `inventory_system.log`
- Review the documentation in `paper/agentic_inventory_paper.docx`
- Ensure all data files are properly formatted

## Troubleshooting

1. **Import Errors**: Run `pip install -r requirements.txt`
2. **Data Errors**: Verify CSV file formats and headers
3. **Permission Errors**: Check file and directory permissions
4. **Dashboard Issues**: Try `streamlit run ui/app.py` directly
"""
    
    with open('QUICK_START.md', 'w') as f:
        f.write(guide_content)
    print("✅ Created QUICK_START.md")

def main():
    """Main setup function"""
    print("🤖 Agentic Inventory Management System - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    install_requirements()
    
    # Setup environment
    setup_environment()
    
    # Create quick start guide
    create_quick_start_guide()
    
    # Verify installation
    if verify_installation():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Review and edit .env file with your API keys")
        print("2. Run 'python main.py' to start the system")
        print("3. Or run 'python main.py dashboard' to launch the web interface")
        print("4. Check QUICK_START.md for detailed instructions")
    else:
        print("\n⚠️  Setup completed with warnings")
        print("Please check the error messages above and resolve any issues")

if __name__ == "__main__":
    main()
