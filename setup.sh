#!/bin/bash

# Flink ML Inference SDK Setup Script
set -e

echo "🚀 Flink ML Inference SDK Setup"
echo "================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
check_prerequisites() {
    echo "📋 Checking prerequisites..."

    # Check Java
    if command -v java &> /dev/null; then
        JAVA_VERSION=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}' | cut -d. -f1)
        if [ "$JAVA_VERSION" -ge 11 ]; then
            echo -e "${GREEN}✓${NC} Java $JAVA_VERSION found"
        else
            echo -e "${RED}✗${NC} Java 11 or higher required (found Java $JAVA_VERSION)"
            exit 1
        fi
    else
        echo -e "${RED}✗${NC} Java not found. Please install JDK 11 or higher"
        exit 1
    fi

    # Check Maven
    if command -v mvn &> /dev/null; then
        MVN_VERSION=$(mvn -version | head -n 1 | awk '{print $3}')
        echo -e "${GREEN}✓${NC} Maven $MVN_VERSION found"
    else
        echo -e "${RED}✗${NC} Maven not found. Please install Maven 3.6+"
        exit 1
    fi

    # Check Git
    if command -v git &> /dev/null; then
        echo -e "${GREEN}✓${NC} Git found"
    else
        echo -e "${RED}✗${NC} Git not found. Please install Git"
        exit 1
    fi
}

# Create project structure
create_structure() {
    echo ""
    echo "📁 Creating project structure..."

    # Create all module directories
    modules=(
        "flink-ml-inference-core"
        "flink-ml-inference-onnx"
        "flink-ml-inference-tensorflow"
        "flink-ml-inference-pytorch"
        "flink-ml-inference-xgboost"
        "flink-ml-inference-pmml"
        "flink-ml-inference-remote"
        "flink-ml-inference-examples"
    )

    for module in "${modules[@]}"; do
        mkdir -p "$module/src/main/java/com/flinkml/inference"
        mkdir -p "$module/src/main/resources"
        mkdir -p "$module/src/test/java/com/flinkml/inference"
        mkdir -p "$module/src/test/resources"
    done

    # Create additional directories
    mkdir -p .github/workflows
    mkdir -p docs
    mkdir -p models
    mkdir -p scripts

    echo -e "${GREEN}✓${NC} Project structure created"
}

# Initialize Git repository
init_git() {
    echo ""
    echo "🔧 Initializing Git repository..."

    if [ ! -d ".git" ]; then
        git init

        # Create .gitignore
        cat > .gitignore << 'EOF'
# Maven
target/
pom.xml.tag
pom.xml.releaseBackup
pom.xml.versionsBackup
pom.xml.next
release.properties
dependency-reduced-pom.xml
buildNumber.properties
.mvn/timing.properties

# IDE
.idea/
*.iml
*.ipr
*.iws
.project
.classpath
.settings/
.vscode/
*.swp
*.bak
*~

# OS
.DS_Store
Thumbs.db

# Models
models/*.onnx
models/*.pb
models/*.pt
models/*.pth
!models/.gitkeep

# Logs
*.log
logs/

# Temp
*.tmp
temp/

# Secrets
*.env
secrets/
EOF

        # Create .gitkeep for models directory
        touch models/.gitkeep

        git add .
        git commit -m "Initial commit: Flink ML Inference SDK"

        echo -e "${GREEN}✓${NC} Git repository initialized"
    else
        echo -e "${YELLOW}⚠${NC} Git repository already exists"
    fi
}

# Build project
build_project() {
    echo ""
    echo "🔨 Building project..."
    echo ""

    read -p "Skip tests for faster build? (y/n): " skip_tests

    if [ "$skip_tests" = "y" ] || [ "$skip_tests" = "Y" ]; then
        mvn clean install -DskipTests
    else
        mvn clean install
    fi

    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓${NC} Build successful!"
    else
        echo ""
        echo -e "${RED}✗${NC} Build failed"
        exit 1
    fi
}

# Setup GitHub Actions
setup_github_actions() {
    echo ""
    echo "⚙️  Setting up GitHub Actions..."

    if [ ! -f ".github/workflows/ci.yml" ]; then
        echo -e "${YELLOW}⚠${NC} GitHub Actions workflow not found at .github/workflows/ci.yml"
        echo "   Please create it manually using the provided ci.yml template"
    else
        echo -e "${GREEN}✓${NC} GitHub Actions workflow configured"
    fi
}

# Display next steps
show_next_steps() {
    echo ""
    echo "✅ Setup Complete!"
    echo "=================="
    echo ""
    echo "📚 Next Steps:"
    echo ""
    echo "1. Review the implementation:"
    echo "   cat IMPLEMENTATION_CHECKLIST.md"
    echo ""
    echo "2. Export your ML model to ONNX:"
    echo "   # PyTorch"
    echo "   torch.onnx.export(model, dummy_input, 'model.onnx')"
    echo ""
    echo "3. Run the fraud detection example:"
    echo "   mvn exec:java -pl flink-ml-inference-examples \\"
    echo "     -Dexec.mainClass='com.flinkml.inference.examples.fraud.FraudDetectionExample'"
    echo ""
    echo "4. Configure GitHub Secrets for CI/CD:"
    echo "   - OSSRH_USERNAME"
    echo "   - OSSRH_PASSWORD"
    echo "   - GPG_PRIVATE_KEY"
    echo "   - GPG_PASSPHRASE"
    echo ""
    echo "5. Create a release:"
    echo "   git tag -a v1.0.0 -m 'Release v1.0.0'"
    echo "   git push origin v1.0.0"
    echo ""
    echo "📖 Documentation:"
    echo "   - README.md - General usage"
    echo "   - BUILD.md - Build instructions"
    echo "   - IMPLEMENTATION_CHECKLIST.md - Development status"
    echo ""
    echo "🎯 Quick test with ONNX model:"
    echo "   1. Place your model in: models/your_model.onnx"
    echo "   2. Update example code with your model path"
    echo "   3. Run the example"
    echo ""
}

# Main execution
main() {
    check_prerequisites

    read -p "Create project structure? (y/n): " create_struct
    if [ "$create_struct" = "y" ] || [ "$create_struct" = "Y" ]; then
        create_structure
    fi

    read -p "Initialize Git repository? (y/n): " init_git_repo
    if [ "$init_git_repo" = "y" ] || [ "$init_git_repo" = "Y" ]; then
        init_git
    fi

    read -p "Build project now? (y/n): " build
    if [ "$build" = "y" ] || [ "$build" = "Y" ]; then
        build_project
    fi

    setup_github_actions
    show_next_steps
}

# Run main
main