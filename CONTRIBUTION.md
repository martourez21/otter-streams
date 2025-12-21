#  Contributing to Otter-Streams

Thank you for your interest in contributing to Otter-Streams! We're excited to have you join our community of developers building the future of real-time ML inference for Apache Flink.

## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Building the Project](#building-the-project)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Documentation](#documentation)
- [Community](#community)
- [Release Process](#release-process)

## 📜 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

##  Getting Started

### Ways to Contribute
- 🐛 **Report bugs** - Use GitHub Issues with detailed reproduction steps
- 💡 **Suggest features** - Start a discussion or create an issue
- 📚 **Improve documentation** - Fix typos, add examples, improve clarity
- 🔧 **Fix issues** - Check the "good first issue" label
- 🧪 **Add tests** - Improve test coverage
- 🎨 **Code contributions** - Implement new features or bug fixes

### First Time Contributors
Look for issues labeled:
- `good first issue` - Great for newcomers
- `help wanted` - Community contributions needed
- `documentation` - Non-code contributions welcome

## 💻 Development Environment

### Prerequisites
- **Java 11+** (JDK 11 or 17 recommended)
- **Maven 3.6+**
- **Git**
- **Docker** (optional, for integration tests)

### Setup Steps
1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone:
   git clone https://github.com/martourez21/otter-streams.git
   cd otter-streams
   ```

2. **Set up upstream remote**
   ```bash
   git remote add upstream https://github.com/martourez21/otter-streams.git
   ```

3. **Install dependencies**
   ```bash
   # Build all modules
   mvn clean install -DskipTests
   ```

## 🏗️ Project Structure

```
otter-streams/
├── ml-inference-core/          # Core inference engine
├── otter-stream-onnx/         # ONNX Runtime integration
├── otter-stream-tensorflow/   # TensorFlow SavedModel support
├── otter-stream-pytorch/      # PyTorch model inference
├── otter-stream-xgboost/      # XGBoost integration
├── otter-stream-pmml/         # PMML model support
├── otter-stream-remote/       # Remote inference service
├── otter-stream-examples/     # Usage examples
├── docs/                      # Documentation
└── .github/workflows/         # CI/CD pipelines
```

## 🔨 Building the Project

### Build Options
```bash
# Build everything
mvn clean install

# Skip tests
mvn clean install -DskipTests

# Build specific module
cd otter-stream-onnx
mvn clean install

# Build with specific Java version
JAVA_HOME=/path/to/java11 mvn clean install
```

### Common Build Tasks
```bash
# Run all tests
mvn test

# Run tests for specific module
cd otter-stream-onnx
mvn test

# Generate Javadoc
mvn javadoc:aggregate

# Check code style
mvn checkstyle:check

# Run integration tests
mvn verify
```

## 🧪 Testing

### Test Structure
- **Unit Tests**: In `src/test/java/`
- **Integration Tests**: In `src/it/java/`
- **Example Tests**: In `otter-stream-examples/`

### Running Tests
```bash
# Run all tests
mvn test

# Run tests with coverage
mvn test jacoco:report

# Run specific test class
mvn test -Dtest=InferenceEngineTest

# Run tests matching pattern
mvn test -Dtest="*Test"
```

### Writing Tests
- Use JUnit 5
- Follow Arrange-Act-Assert pattern
- Include meaningful test names
- Test both success and failure cases
- Mock external dependencies
- Clean up resources in `@AfterEach`

Example:
```java
@Test
void shouldExecuteModelInferenceSuccessfully() {
    // Arrange
    var model = loadTestModel();
    var input = createTestInput();
    
    // Act
    var result = inferenceEngine.execute(model, input);
    
    // Assert
    assertNotNull(result);
    assertEquals(expectedOutput, result);
}
```

## 🔄 Pull Request Process

### Before You Start
1. **Check for existing issues/PRs** - Avoid duplicates
2. **Discuss major changes** - Start a discussion first
3. **Create a branch** - Use descriptive names:
   ```bash
   git checkout -b feature/add-new-model-support
   git checkout -b fix/memory-leak-in-cache
   ```

### PR Checklist
- [ ] Code compiles without errors
- [ ] All tests pass
- [ ] New tests added for changes
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] Commit messages follow conventions
- [ ] PR description explains changes
- [ ] Linked to relevant issues

### Commit Message Convention
```
<type>: <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting, missing semi-colons, etc.
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat: add support for TensorFlow 2.15 models
fix: resolve memory leak in model cache
docs: update getting started guide
test: add integration tests for remote inference
```

## 📝 Code Style Guidelines

### Java Code Style
- Follow Java 11+ conventions
- Use `var` for local variables when type is obvious
- Use final where appropriate
- Avoid wildcard imports
- Use 4-space indentation
- Keep lines under 120 characters
- Use meaningful names

### Naming Conventions
- Classes: `PascalCase` (e.g., `InferenceEngine`)
- Methods: `camelCase` (e.g., `executeModel`)
- Variables: `camelCase` (e.g., `inputTensor`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_BATCH_SIZE`)
- Packages: `lowercase` (e.g., `com.codedstreams.inference`)

### Error Handling
- Use specific exception types
- Include meaningful error messages
- Log exceptions appropriately
- Clean up resources in finally blocks
- Use try-with-resources for AutoCloseable

### Logging
- Use SLF4J for logging
- Appropriate log levels:
    - ERROR: Unrecoverable errors
    - WARN: Recoverable issues
    - INFO: Important runtime events
    - DEBUG: Detailed debugging
    - TRACE: Very detailed tracing

## 📚 Documentation

### Documentation Types
1. **Javadoc**: All public APIs must have Javadoc
2. **README**: Module-specific documentation
3. **Examples**: Working code examples
4. **Website**: User documentation at [martourez21.github.io/otter-streams](https://martourez21.github.io/otter-streams)

### Writing Javadoc
```java
/**
 * Executes model inference on the provided input data.
 * 
 * @param model The model to execute
 * @param input The input data for inference
 * @return The inference result
 * @throws InferenceException if inference fails
 * @throws IllegalArgumentException if input is invalid
 */
public InferenceResult execute(Model model, InferenceInput input) {
    // implementation
}
```

### Updating Website
Documentation website is in `.github/workflows/templates/`:
- HTML templates for different pages
- Images in `assets/` directory
- Run CI pipeline to deploy updates

## 👥 Community

### Getting Help
- **Discussions**: For questions and ideas
- **Issues**: For bug reports and feature requests
- **Email**: nestorabiawuh@gmail.com

### Communication Channels
- GitHub Discussions: For community conversations
- GitHub Issues: For technical issues
- Email: For direct maintainer contact

### Recognition
All contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in release notes
- Featured in community updates

## 🚢 Release Process

### Versioning
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Release Cycle
1. **Planning**: Issues tagged for next release
2. **Development**: Features implemented and tested
3. **Testing**: Integration and regression testing
4. **Release Candidate**: Tagged and tested
5. **Release**: Final version published
6. **Documentation**: Updated and deployed

### Publishing
Releases are automatically published via GitHub Actions when:
- A release is created on GitHub
- All tests pass
- Documentation is updated

---

## 🎉 Thank You!

Your contributions help make Otter-Streams better for everyone. Whether you're fixing a typo, reporting a bug, or implementing a major feature, every contribution is valued.

**Welcome to the Otter-Streams community!** 

---

*Maintained by [Nestor Martourez](https://github.com/martourez21) at [CodedStreams](https://github.com/martourez21)*
*Apache License 2.0 • © 2025*

---

Need help getting started? Open a discussion or reach out to nestorabiawuh@gmail.com