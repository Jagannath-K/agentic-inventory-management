#!/bin/bash

# Agentic Inventory Management System - Setup and Run Script

echo "🤖 Agentic Inventory Management System"
echo "====================================="

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p models

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Function to display menu
show_menu() {
    echo ""
    echo "Select an option:"
    echo "1. Run single inventory cycle"
    echo "2. Start automated scheduler"
    echo "3. Launch dashboard"
    echo "4. Train prediction models"
    echo "5. Interactive mode"
    echo "6. Exit"
    echo ""
}

# Function to run inventory cycle
run_cycle() {
    echo "Running inventory management cycle..."
    python main.py cycle
}

# Function to start scheduler
start_scheduler() {
    echo "Starting automated scheduler..."
    echo "Press Ctrl+C to stop"
    python main.py scheduler
}

# Function to launch dashboard
launch_dashboard() {
    echo "Launching Streamlit dashboard..."
    echo "Dashboard will open in your browser"
    python main.py dashboard
}

# Function to train models
train_models() {
    echo "Training prediction models..."
    python main.py train
}

# Function to run interactive mode
interactive_mode() {
    echo "Starting interactive mode..."
    python main.py
}

# Main menu loop
while true; do
    show_menu
    read -p "Enter your choice (1-6): " choice
    
    case $choice in
        1)
            run_cycle
            ;;
        2)
            start_scheduler
            ;;
        3)
            launch_dashboard
            ;;
        4)
            train_models
            ;;
        5)
            interactive_mode
            ;;
        6)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid option. Please select 1-6."
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
done
