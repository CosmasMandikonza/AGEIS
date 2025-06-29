echo "=== Aegis Environment Setup ==="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Pull required models
echo "Pulling Granite models..."
ollama pull granite-4.0-tiny
ollama pull granite-guardian-8b

# Create directories
echo "Creating project directories..."
mkdir -p data/compliance_docs
mkdir -p data/processed

# Copy environment template
echo "Setting up environment file..."
cp .env.example .env
echo "⚠️  Please edit .env file with your IBM Watson credentials"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your IBM Watson credentials"
echo "2. Run: python scripts/test_connections.py"
echo "3. Run: python scripts/prepare_data.py"
echo "4. Run: python app.py"