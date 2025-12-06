# Building Flink ML Inference SDK

## Prerequisites

- **JDK 11 or higher** (Java 11, 17 recommended)
- **Apache Maven 3.6+**
- **Git**

## Quick Build

```bash
# Clone the repository
git clone https://github.com/yourusername/flink-ml-inference-sdk.git
cd flink-ml-inference-sdk

# Build all modules
mvn clean install

# Skip tests for faster build
mvn clean install -DskipTests
```

## Module Structure

```
flink-ml-inference-sdk/
├── ml-inference-core/          # Core interfaces and abstractions
├── otter-stream-onnx/          # ONNX Runtime implementation
├── otter-stream-tensorflow/    # TensorFlow implementation
├── otter-stream-pytorch/       # PyTorch implementation
├── otter-streame-xgboost/       # XGBoost implementation
├── otter-stream-pmml/          # PMML implementation
├── otter-stream-remote/        # Remote inference clients
└── otter-stream-examples/      # Usage examples
```

## Building Specific Modules

```bash
# Build only core module
mvn clean install -pl flink-ml-inference-core

# Build core + ONNX
mvn clean install -pl ml-inference-core,otter-stream-onnx -am

# Build with dependencies
mvn clean install -pl otter-stream-onnx -am
```

## Running Tests

```bash
# Run all tests
mvn test

# Run tests for specific module
mvn test -pl ml-inference-core

# Run integration tests
mvn verify

# Skip integration tests
mvn install -DskipITs
```

## Code Coverage

```bash
# Generate coverage report
mvn clean test jacoco:report

# View report at: target/site/jacoco/index.html

# Check coverage thresholds
mvn clean verify -P coverage
```

## Building Distribution

```bash
# Build JAR with dependencies
mvn clean package -P shade

# Build all artifacts (sources, javadocs)
mvn clean package -P release

# Create distribution ZIP
mvn clean install assembly:single
```

## IDE Setup

### IntelliJ IDEA

1. Import project as Maven project
2. File → Project Structure → Project SDK → Select JDK 11+
3. Enable annotation processing
4. Install Lombok plugin (if using Lombok)

### Eclipse

1. Import → Existing Maven Projects
2. Select root directory
3. Right-click project → Maven → Update Project

### VS Code

1. Install Extension Pack for Java
2. Open folder
3. Maven extension will auto-detect project

## GitHub Actions Setup

### Required Secrets

Configure these in GitHub Repository Settings → Secrets:

```bash
# Maven Central Deployment
OSSRH_USERNAME=your-sonatype-username
OSSRH_PASSWORD=your-sonatype-password

# GPG Signing
GPG_PRIVATE_KEY=your-gpg-private-key
GPG_PASSPHRASE=your-gpg-passphrase

# Docker Hub (optional)
DOCKER_USERNAME=your-docker-username
DOCKER_PASSWORD=your-docker-token

# Code Quality (optional)
SONAR_TOKEN=your-sonarqube-token
SONAR_HOST_URL=https://sonarcloud.io
```

### Generating GPG Key

```bash
# Generate key
gpg --gen-key

# List keys
gpg --list-secret-keys --keyid-format=long

# Export private key (for GitHub Secrets)
gpg --armor --export-secret-keys YOUR_KEY_ID

# Export public key (for Maven Central)
gpg --keyserver keyserver.ubuntu.com --send-keys YOUR_KEY_ID
```

### GitHub Actions Workflows

The project includes these workflows:

1. **CI Pipeline** (`.github/workflows/ci.yml`)
    - Builds on Java 11 and 17
    - Runs tests
    - Code coverage with Codecov
    - Code quality with SonarQube
    - Security scanning with OWASP

2. **Snapshot Deployment** (triggered on `develop` branch)
    - Deploys SNAPSHOT versions to OSSRH

3. **Release Deployment** (triggered on version tags)
    - Builds and signs artifacts
    - Deploys to Maven Central
    - Creates GitHub Release
    - Builds Docker images

### Triggering Releases

```bash
# Create and push version tag
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# GitHub Actions will automatically:
# 1. Run full test suite
# 2. Build and sign artifacts
# 3. Deploy to Maven Central
# 4. Create GitHub Release
# 5. Build Docker image
```

## Publishing to Maven Central

### Setup (One-time)

1. **Create Sonatype JIRA account**
    - Go to https://issues.sonatype.org
    - Create account and request namespace

2. **Generate GPG key** (see above)

3. **Configure Maven settings.xml**

```xml
<settings>
  <servers>
    <server>
      <id>ossrh</id>
      <username>${env.OSSRH_USERNAME}</username>
      <password>${env.OSSRH_PASSWORD}</password>
    </server>
  </servers>
  <profiles>
    <profile>
      <id>ossrh</id>
      <activation>
        <activeByDefault>true</activeByDefault>
      </activation>
      <properties>
        <gpg.executable>gpg</gpg.executable>
        <gpg.passphrase>${env.GPG_PASSPHRASE}</gpg.passphrase>
      </properties>
    </profile>
  </profiles>
</settings>
```

### Manual Release

```bash
# Set version
mvn versions:set -DnewVersion=1.0.0

# Deploy to staging
mvn clean deploy -P release

# Release (if not auto-release)
mvn nexus-staging:release
```

## Docker Image

```bash
# Build Docker image
docker build -t flink-ml-inference:latest .

# Run with Flink
docker run -v /path/to/models:/models \
  flink-ml-inference:latest
```

## Troubleshooting

### Build Fails with "Cannot resolve dependencies"

```bash
# Clear Maven cache
rm -rf ~/.m2/repository

# Rebuild
mvn clean install -U
```

### Tests Fail

```bash
# Run single test
mvn test -Dtest=YourTestClass

# Run with debug logging
mvn test -X

# Skip flaky tests
mvn test -Dsurefire.excludes=FlakyTest.java
```

### GPG Signing Issues

```bash
# Test GPG signing
mvn clean verify -P release -Dgpg.skip=false

# If "gpg: signing failed: Inappropriate ioctl for device"
export GPG_TTY=$(tty)
```

### Out of Memory During Build

```bash
# Increase Maven memory
export MAVEN_OPTS="-Xmx2g -XX:MaxPermSize=512m"

# Or in .mvn/jvm.config
echo "-Xmx2g" > .mvn/jvm.config
```

## Performance Testing

```bash
# Run performance tests
mvn verify -P performance-tests

# With JMH benchmarks
mvn clean install
java -jar flink-ml-inference-benchmarks/target/benchmarks.jar
```

## Continuous Integration

### Local CI Simulation

```bash
# Install act (GitHub Actions locally)
brew install act

# Run CI locally
act -P ubuntu-latest=nektos/act-environments-ubuntu:18.04
```

### Branch Protection Rules

Recommended GitHub branch protection for `main`:

- ✅ Require pull request reviews (1 approval)
- ✅ Require status checks to pass (CI build)
- ✅ Require branches to be up to date
- ✅ Include administrators
- ✅ Require linear history

## Documentation

```bash
# Generate Javadocs
mvn javadoc:aggregate

# View at: target/site/apidocs/index.html

# Build site documentation
mvn site

# Deploy to GitHub Pages
mvn site:deploy
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Support

- **Issues**: https://github.com/yourusername/flink-ml-inference-sdk/issues
- **Discussions**: https://github.com/yourusername/flink-ml-inference-sdk/discussions
- **Stack Overflow**: Tag with `otter-sstreams`