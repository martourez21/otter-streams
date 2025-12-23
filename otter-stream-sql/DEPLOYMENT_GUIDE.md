# Otter Stream SQL - Complete Deployment Guide

This guide covers deploying Otter Stream SQL on both **self-managed Flink clusters** and **Confluent Cloud for Apache Flink**.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Building the Project](#building-the-project)
3. [Self-Managed Flink Deployment](#self-managed-flink-deployment)
4. [Confluent Cloud Deployment](#confluent-cloud-deployment)
5. [AWS Kinesis Data Analytics](#aws-kinesis-data-analytics)
6. [Configuration Examples](#configuration-examples)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

- Java 11+
- Maven 3.6+
- Apache Flink 1.17.0
- TensorFlow models (SavedModel or GraphDef format)

### Storage Access

Ensure you have credentials for:
- AWS S3 (if using S3 models)
- MinIO (if using MinIO storage)
- HTTP endpoints (if using remote models)

---

## Building the Project

### 1. Clone the Repository

```bash
git checkout -b feature/sql-integration
cd otter-streams
```

### 2. Build All Modules

```bash
mvn clean install -DskipTests
```

### 3. Build SQL Module Only

```bash
cd otter-stream-sql
mvn clean package
```

The shaded JAR will be in `target/otter-stream-sql-1.0.15.jar`

---

## Self-Managed Flink Deployment

### Option A: Deploy via Flink lib Directory

**Step 1: Copy JAR to Flink**

```bash
# Copy to Flink lib directory
cp otter-stream-sql/target/otter-stream-sql-1.0.15.jar $FLINK_HOME/lib/

# Restart Flink cluster
$FLINK_HOME/bin/stop-cluster.sh
$FLINK_HOME/bin/start-cluster.sh
```

**Step 2: Use in SQL Client**

```bash
# Start SQL client
$FLINK_HOME/bin/sql-client.sh
```

```sql
-- Register UDF
CREATE TEMPORARY FUNCTION ML_PREDICT AS 
    'com.codedstreams.otterstream.sql.udf.MLPredictScalarFunction';

-- Use in queries
SELECT 
    transaction_id,
    ML_PREDICT('fraud-model', 
               JSON_OBJECT('amount', amount, 'merchant', merchant)) AS score
FROM transactions;
```

### Option B: Submit with Job

**Step 1: Create Job JAR**

Include Otter Stream SQL as dependency:

```xml
<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>otter-stream-sql</artifactId>
    <version>1.0.15</version>
</dependency>
```

**Step 2: Submit Job**

```bash
flink run \
    -c com.mycompany.MyFlinkJob \
    -C file:///path/to/otter-stream-sql-1.0.15.jar \
    my-job.jar
```

### Option C: Kubernetes Deployment

**flink-configuration-configmap.yaml**:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: flink-config
data:
  flink-conf.yaml: |
    jobmanager.rpc.address: flink-jobmanager
    taskmanager.numberOfTaskSlots: 4
    parallelism.default: 2
```

**Dockerfile**:

```dockerfile
FROM flink:1.17.0-scala_2.12-java11

# Copy Otter Stream SQL JAR
COPY otter-stream-sql-1.0.15.jar /opt/flink/lib/

# Copy your job JAR
COPY my-job.jar /opt/flink/usrlib/

USER flink
```

**Deploy**:

```bash
# Build image
docker build -t my-flink-job:latest .

# Push to registry
docker push my-registry/my-flink-job:latest

# Apply Kubernetes manifests
kubectl apply -f flink-configuration-configmap.yaml
kubectl apply -f jobmanager-service.yaml
kubectl apply -f jobmanager-deployment.yaml
kubectl apply -f taskmanager-deployment.yaml
```

---

## Confluent Cloud Deployment

### Prerequisites

1. Confluent Cloud account
2. Flink compute pool created
3. Kafka topics configured

### Step 1: Upload JAR to Confluent

**Via Confluent CLI**:

```bash
# Install Confluent CLI
curl -sL --http1.1 https://cnfl.io/cli | sh -s -- latest

# Login
confluent login

# Upload JAR
confluent flink artifact create \
    --cloud aws \
    --region us-east-1 \
    --artifact-file otter-stream-sql-1.0.15.jar
```

**Via Confluent Cloud Console**:

1. Navigate to **Flink** → **Artifacts**
2. Click **Upload Artifact**
3. Select `otter-stream-sql-1.0.15.jar`
4. Wait for upload to complete

### Step 2: Create Flink SQL Statement

**Via SQL Workspace**:

```sql
-- Register artifact (automatic in Confluent Cloud)

-- Register function
CREATE TEMPORARY FUNCTION ML_PREDICT AS 
    'com.codedstreams.otterstream.sql.udf.MLPredictScalarFunction'
    USING JAR 'otter-stream-sql-1.0.15.jar';

-- Create source table
CREATE TABLE transactions (
    transaction_id STRING,
    user_id STRING,
    amount DOUBLE,
    merchant STRING,
    merchant_category STRING,
    transaction_time TIMESTAMP(3),
    WATERMARK FOR transaction_time AS transaction_time - INTERVAL '5' SECOND
) WITH (
    'connector' = 'confluent',
    'kafka.topic' = 'transactions',
    'value.format' = 'json'
);

-- Create sink table
CREATE TABLE fraud_alerts (
    transaction_id STRING,
    fraud_score DOUBLE,
    alert_time TIMESTAMP(3)
) WITH (
    'connector' = 'confluent',
    'kafka.topic' = 'fraud-alerts',
    'value.format' = 'json'
);

-- Run inference
INSERT INTO fraud_alerts
SELECT 
    transaction_id,
    ML_PREDICT('fraud-detector', 
        JSON_OBJECT(
            'amount', amount,
            'merchant_category', merchant_category
        )
    ) AS fraud_score,
    CURRENT_TIMESTAMP AS alert_time
FROM transactions
WHERE ML_PREDICT('fraud-detector', 
    JSON_OBJECT('amount', amount, 'merchant_category', merchant_category)
) > 0.7;
```

### Step 3: Model Storage Configuration

**Option A: S3 with IAM Role**

```sql
-- Create connector with S3 access
CREATE TABLE ml_predictions (
    ...
) WITH (
    'connector' = 'ml-inference',
    'model.path' = 's3://my-confluent-bucket/models/fraud/',
    'model.format' = 'tensorflow-savedmodel',
    'model.s3.region' = 'us-east-1'
    -- IAM role attached to Flink compute pool
);
```

**Option B: S3 with Access Keys**

```sql
-- Using Confluent Secrets
CREATE TABLE ml_predictions (
    ...
) WITH (
    'connector' = 'ml-inference',
    'model.path' = 's3://my-bucket/models/',
    'model.s3.access-key' = '${AWS_ACCESS_KEY}',
    'model.s3.secret-key' = '${AWS_SECRET_KEY}',
    'model.s3.region' = 'us-east-1'
);
```

**Option C: HTTP Model Server**

```sql
CREATE TABLE ml_predictions (
    ...
) WITH (
    'connector' = 'ml-inference',
    'model.path' = 'https://model-server.example.com/models/v1/',
    'model.http.auth-token' = '${MODEL_SERVER_TOKEN}'
);
```

### Step 4: Start Statement

**Via CLI**:

```bash
confluent flink statement create \
    --sql "INSERT INTO fraud_alerts SELECT ..." \
    --compute-pool <pool-id> \
    --service-account <sa-id>
```

**Via Console**:

1. Navigate to **SQL Workspace**
2. Paste SQL statements
3. Click **Run**
4. Monitor execution

---

## AWS Kinesis Data Analytics

### Step 1: Upload to S3

```bash
# Upload JAR
aws s3 cp otter-stream-sql-1.0.15.jar \
    s3://my-kda-bucket/jars/otter-stream-sql-1.0.15.jar
```

### Step 2: Create KDA Application

**Via AWS Console**:

1. Go to **Kinesis Data Analytics**
2. Create application (Apache Flink 1.17)
3. Under **Application Configuration**:
    - Add JAR: `s3://my-kda-bucket/jars/otter-stream-sql-1.0.15.jar`

**Via CloudFormation**:

```yaml
Resources:
  FlinkApplication:
    Type: AWS::KinesisAnalyticsV2::Application
    Properties:
      ApplicationName: fraud-detection-ml
      RuntimeEnvironment: FLINK-1_17
      ServiceExecutionRole: !GetAtt FlinkExecutionRole.Arn
      ApplicationConfiguration:
        FlinkApplicationConfiguration:
          MonitoringConfiguration:
            ConfigurationType: CUSTOM
            MetricsLevel: APPLICATION
            LogLevel: INFO
        ApplicationCodeConfiguration:
          CodeContent:
            S3ContentLocation:
              BucketARN: !Sub 'arn:aws:s3:::my-kda-bucket'
              FileKey: 'jars/otter-stream-sql-1.0.15.jar'
          CodeContentType: ZIPFILE
```

### Step 3: SQL Script

Create `sql-script.sql`:

```sql
-- Register function
CREATE TEMPORARY FUNCTION ML_PREDICT AS 
    'com.codedstreams.otterstream.sql.udf.MLPredictScalarFunction';

-- Source from Kinesis
CREATE TABLE transactions (
    transaction_id VARCHAR,
    amount DOUBLE,
    merchant VARCHAR,
    event_time TIMESTAMP(3),
    WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
) WITH (
    'connector' = 'kinesis',
    'stream' = 'transactions-stream',
    'aws.region' = 'us-east-1',
    'format' = 'json'
);

-- Sink to Kinesis
CREATE TABLE alerts (
    transaction_id VARCHAR,
    fraud_score DOUBLE,
    alert_time TIMESTAMP(3)
) WITH (
    'connector' = 'kinesis',
    'stream' = 'fraud-alerts-stream',
    'aws.region' = 'us-east-1',
    'format' = 'json'
);

-- Inference query
INSERT INTO alerts
SELECT 
    transaction_id,
    ML_PREDICT('fraud-model',
        JSON_OBJECT('amount', amount, 'merchant', merchant)
    ) AS fraud_score,
    CURRENT_TIMESTAMP
FROM transactions;
```

### Step 4: Deploy

```bash
# Upload SQL script
aws s3 cp sql-script.sql s3://my-kda-bucket/scripts/

# Start application
aws kinesisanalyticsv2 start-application \
    --application-name fraud-detection-ml \
    --run-configuration '{
        "SqlRunConfigurations": [{
            "InputId": "1.1",
            "InputStartingPositionConfiguration": {
                "InputStartingPosition": "NOW"
            }
        }]
    }'
```

---

## Configuration Examples

### Example 1: S3 Model with Local Cache

```sql
CREATE TABLE predictions (
    features STRING,
    prediction DOUBLE
) WITH (
    'connector' = 'ml-inference',
    'model.name' = 'fraud-detector',
    'model.path' = 's3://my-models/fraud-v2/',
    'model.format' = 'tensorflow-savedmodel',
    'model.s3.region' = 'us-east-1',
    'cache.enabled' = 'true',
    'cache.max-size' = '10',
    'cache.ttl-minutes' = '60',
    'batch.size' = '32',
    'batch.timeout-ms' = '100'
);
```

### Example 2: HTTP Model with Authentication

```sql
CREATE TABLE predictions (
    features STRING,
    prediction DOUBLE
) WITH (
    'connector' = 'ml-inference',
    'model.name' = 'recommendation',
    'model.path' = 'https://ml-api.company.com/models/recommender/',
    'model.format' = 'tensorflow-savedmodel',
    'model.http.auth-token' = 'Bearer ${API_TOKEN}',
    'async.enabled' = 'true',
    'async.timeout-ms' = '3000'
);
```

### Example 3: MinIO with Custom Endpoint

```sql
CREATE TABLE predictions (
    features STRING,
    prediction DOUBLE
) WITH (
    'connector' = 'ml-inference',
    'model.name' = 'sentiment',
    'model.path' = 'minio://minio.internal.com/ml-models/sentiment/',
    'model.format' = 'tensorflow-graphdef',
    'model.minio.access-key' = '${MINIO_ACCESS_KEY}',
    'model.minio.secret-key' = '${MINIO_SECRET_KEY}'
);
```

### Example 4: Local Model (Testing)

```sql
CREATE TABLE predictions (
    features STRING,
    prediction DOUBLE
) WITH (
    'connector' = 'ml-inference',
    'model.name' = 'test-model',
    'model.path' = 'file:///opt/flink/models/test-model/',
    'model.format' = 'tensorflow-savedmodel'
);
```

---

## Troubleshooting

### Issue 1: Model Not Found

**Error**:
```
Model not found in cache: fraud-detector
```

**Solution**:
1. Verify model path is accessible
2. Check S3/MinIO credentials
3. Ensure model format is correct
4. Register model before use:

```sql
-- Pre-load model (if using registry)
CALL register_model('fraud-detector', 's3://bucket/model/', 'tensorflow-savedmodel');
```

### Issue 2: ClassNotFoundException

**Error**:
```
ClassNotFoundException: com.codedstreams.otterstream.sql.udf.MLPredictScalarFunction
```

**Solution**:
1. Verify JAR is in Flink lib directory
2. Check JAR is included in job submission
3. Restart Flink cluster after adding JAR

### Issue 3: TensorFlow Native Library Error

**Error**:
```
UnsatisfiedLinkError: no tensorflow_jni in java.library.path
```

**Solution**:
1. Ensure TensorFlow native libraries are available
2. Use `tensorflow-core-platform` dependency (includes natives)
3. Set `LD_LIBRARY_PATH` if using custom TensorFlow

### Issue 4: S3 Access Denied

**Error**:
```
Access Denied (Service: S3, Status Code: 403)
```

**Solution**:
1. Verify IAM role/credentials
2. Check S3 bucket permissions
3. Ensure region is correct

**Test access**:
```bash
aws s3 ls s3://my-bucket/models/ --region us-east-1
```

### Issue 5: Inference Timeout

**Error**:
```
Inference timeout after 5000ms
```

**Solution**:
1. Increase timeout:
```sql
'async.timeout-ms' = '10000'
```
2. Enable async mode:
```sql
'async.enabled' = 'true'
```
3. Reduce batch size if batching

### Issue 6: Memory Issues

**Error**:
```
OutOfMemoryError: Java heap space
```

**Solution**:
1. Increase TaskManager memory:
```yaml
taskmanager.memory.process.size: 4g
```
2. Reduce cache size:
```sql
'cache.max-size' = '5'
```
3. Reduce batch size

---

## Performance Tuning

### For High Throughput

```sql
'batch.size' = '64',
'batch.timeout-ms' = '200',
'cache.enabled' = 'true',
'async.enabled' = 'false'  -- Sync for throughput
```

### For Low Latency

```sql
'batch.size' = '1',
'async.enabled' = 'true',
'async.timeout-ms' = '2000',
'cache.enabled' = 'true'
```

### For Remote Models

```sql
'async.enabled' = 'true',
'async.timeout-ms' = '5000',
'retry.max-attempts' = '3',
'retry.backoff-ms' = '100'
```

---

## Monitoring

### Flink Metrics

Available metrics:
- `ml_inference.requests_total`
- `ml_inference.success_total`
- `ml_inference.failures_total`
- `ml_inference.latency_ms`
- `ml_inference.cache_hits`

### View in Flink UI

1. Navigate to **Task Managers**
2. Select task
3. Go to **Metrics** tab
4. Filter for `ml_inference`

---

## Support

- GitHub Issues: https://github.com/martourez21/otter-streams/issues
- Email: nestorabiawuh@gmail.com
- Documentation: Full README in `otter-stream-sql/README.md`

---

**Production Checklist**:

- [ ] JAR uploaded/deployed
- [ ] Model accessible from all TaskManagers
- [ ] Credentials configured
- [ ] Functions registered
- [ ] Test queries validated
- [ ] Monitoring configured
- [ ] Error handling tested
- [ ] Performance tuned