#!/bin/bash

# Check if the virtual environment directory exists
if [ ! -d "myenv" ]; then
    # The virtual environment does not exist, so create it using python3
    echo "Creating a new virtual environment named myenv..."
    python3 -m venv myenv
fi

# Activate the virtual environment
echo "Activating the virtual environment..."
source myenv/bin/activate

# Install required Python packages from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip3 install -r requirements.txt

echo "Environment setup is complete."

# Define the project's data directory relative to the script location
DATA_DIR="./data"
# Assuming the variable DATA_DIR is already set to the desired directory
ZIP_FILE="./data.zip"

# Check if the ZIP file exists
if [ ! -f "$ZIP_FILE" ]; then
    echo "Data zip file does not exist. Downloading..."
    gdown "1xAH5thJeGNs-FB4JDMVhldWyp8t-4jNi" -O "$ZIP_FILE"
    echo "Unzipping the data..."
    unzip "$ZIP_FILE" -d "./"
    echo "Data has been downloaded and extracted."
fi

rm data.zip
echo "Your Data is Now Available"

echo "\033[32mYou can start the virtual env with the following command\033[0m"
echo "\033[32msource myenv/bin/activate\033[0m"