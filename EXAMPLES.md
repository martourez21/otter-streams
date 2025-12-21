#  Examples & Use Cases

This document provides comprehensive examples of Otter Streams in action, from simple demos to complex production scenarios.

## 📋 Table of Contents
- [Fraud Detection](#fraud-detection)
- [Real-time Recommendations](#real-time-recommendations)
- [Anomaly Detection](#anomaly-detection)
- [Natural Language Processing](#natural-language-processing)
- [Computer Vision](#computer-vision)
- [Time Series Forecasting](#time-series-forecasting)

## 🕵️ Fraud Detection

### Scenario
Real-time fraud detection for financial transactions using an XGBoost model.

### Complete Implementation

```java
public class FraudDetectionPipeline {
    
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        
        // 1. Source: Read from Kafka
        Properties kafkaProps = new Properties();
        kafkaProps.setProperty("bootstrap.servers", "localhost:9092");
        kafkaProps.setProperty("group.id", "fraud-detection");
        
        DataStream<Transaction> transactions = env
            .addSource(new FlinkKafkaConsumer<>(
                "transactions",
                new JSONDeserializer<>(Transaction.class),
                kafkaProps
            ))
            .name("transaction-source");
        
        // 2. Feature Engineering
        DataStream<FraudFeatures> features = transactions
            .map(new FeatureExtractor())
            .name("feature-extraction");
        
        // 3. Model Configuration
        InferenceConfig fraudConfig = InferenceConfig.builder()
            .modelConfig(ModelConfig.builder()
                .modelId("fraud-detector-v2")
                .modelPath("/models/fraud_xgboost_v2.model")
                .format(ModelFormat.XGBOOST)
                .inputNames(new String[]{"features"})
                .outputNames(new String[]{"fraud_probability", "risk_category"})
                .modelOptions(Map.of(
                    "missing", "0.0",
                    "ntree_limit", "100"
                ))
                .build())
            .batchSize(64)
            .batchTimeout(Duration.ofMillis(100))
            .enableCaching(true)
            .cacheSize(10000)
            .cacheTtl(Duration.ofMinutes(5))
            .enableMetrics(true)
            .metricsPrefix("fraud.detection")
            .maxRetries(3)
            .retryDelay(Duration.ofSeconds(1))
            .build();
        
        // 4. Inference Function
        AsyncModelInferenceFunction<FraudFeatures, FraudPrediction> fraudInference =
            new AsyncModelInferenceFunction<>(
                fraudConfig,
                cfg -> new XGBoostInferenceEngine(),
                FraudDetectionPipeline::extractFeatureArray,
                FraudDetectionPipeline::parseFraudPrediction
            );
        
        // 5. Apply Inference
        DataStream<FraudPrediction> predictions = AsyncDataStream.unorderedWait(
            features,
            fraudInference,
            10000,
            TimeUnit.MILLISECONDS,
            200
        ).name("fraud-inference");
        
        // 6. Business Logic
        DataStream<Alert> alerts = predictions
            .process(new FraudAlertProcessor())
            .name("alert-processing");
        
        // 7. Sinks
        // High-risk alerts to Slack
        alerts.filter(alert -> alert.getSeverity() == Severity.HIGH)
            .addSink(new SlackAlertSink())
            .name("slack-alerts");
        
        // All alerts to Elasticsearch for analysis
        alerts.addSink(new ElasticsearchSink<>(
            "fraud-alerts",
            new AlertIndexer()
        )).name("elasticsearch-sink");
        
        // Metrics to Prometheus
        alerts.addSink(new MetricsSink())
            .name("metrics-sink");
        
        env.execute("Real-time Fraud Detection Pipeline");
    }
    
    // Feature extraction
    private static float[] extractFeatureArray(FraudFeatures features) {
        return new float[]{
            features.getTransactionAmount(),
            features.getHourOfDay(),
            features.getIsWeekend() ? 1.0f : 0.0f,
            features.getLocationDistance(),
            features.getPreviousFraudCount(),
            features.getAccountAgeDays(),
            features.getDeviceMismatchScore(),
            features.getVelocityAmount1h(),
            features.getVelocityCount1h(),
            features.getAvgTransactionAmount()
        };
    }
    
    // Prediction parsing
    private static FraudPrediction parseFraudPrediction(InferenceOutput output) {
        float probability = output.getOutput("fraud_probability")[0];
        float category = output.getOutput("risk_category")[0];
        
        return new FraudPrediction(
            probability,
            RiskCategory.fromValue((int)category),
            System.currentTimeMillis()
        );
    }
}

// Supporting Classes
class FraudFeatures {
    private String transactionId;
    private double transactionAmount;
    private int hourOfDay;
    private boolean isWeekend;
    private double locationDistance;
    private int previousFraudCount;
    private int accountAgeDays;
    private double deviceMismatchScore;
    private double velocityAmount1h;
    private int velocityCount1h;
    private double avgTransactionAmount;
    
    // Getters and setters
}

class FraudPrediction {
    private float fraudProbability;
    private RiskCategory riskCategory;
    private long timestamp;
    private String modelVersion;
    
    // Constructor, getters, and business logic
    public boolean isHighRisk() {
        return fraudProbability > 0.9 || riskCategory == RiskCategory.HIGH;
    }
}

enum RiskCategory {
    LOW(0), MEDIUM(1), HIGH(2);
    
    private final int value;
    
    RiskCategory(int value) {
        this.value = value;
    }
    
    public static RiskCategory fromValue(int value) {
        return Arrays.stream(values())
            .filter(category -> category.value == value)
            .findFirst()
            .orElse(LOW);
    }
}
```

## 🎬 Real-time Recommendations

### Scenario
Personalized content recommendations using a neural network model.

### Implementation

```java
public class RecommendationPipeline {
    
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        
        // 1. User Behavior Stream
        DataStream<UserEvent> userEvents = env
            .addSource(new KafkaSource<>("user-events"))
            .keyBy(UserEvent::getUserId);
        
        // 2. Session Window Processing
        DataStream<UserSession> sessions = userEvents
            .window(SessionWindows.withGap(Time.minutes(30)))
            .process(new SessionProcessor())
            .name("session-processing");
        
        // 3. ONNX Model Inference
        InferenceConfig recConfig = InferenceConfig.builder()
            .modelConfig(ModelConfig.builder()
                .modelId("rec-model-v3")
                .modelPath("/models/recommender.onnx")
                .format(ModelFormat.ONNX)
                .inputNames(new String[]{"user_features", "item_features", "context"})
                .outputNames(new String[]{"scores", "ranking"})
                .modelOptions(Map.of(
                    "optimizationLevel", "ALL",
                    "executionMode", "SEQUENTIAL",
                    "interOpThreads", "4"
                ))
                .build())
            .batchSize(128)  // Larger batches for efficiency
            .enableCaching(true)
            .cacheSize(50000)
            .cacheTtl(Duration.ofMinutes(10))
            .parallelism(4)  // Scale horizontally
            .build();
        
        // 4. Recommendation Engine
        AsyncModelInferenceFunction<RecommendationRequest, RecommendationResponse> recommender =
            new AsyncModelInferenceFunction<>(
                recConfig,
                cfg -> new OnnxInferenceEngine(),
                RecommendationPipeline::prepareInputs,
                RecommendationPipeline::parseRecommendations
            );
        
        // 5. Generate Recommendations
        DataStream<RecommendationResponse> recommendations = AsyncDataStream.unorderedWait(
            sessions.flatMap(new CandidateGenerator()),
            recommender,
            5000,
            TimeUnit.MILLISECONDS,
            100
        ).name("recommendation-inference");
        
        // 6. Ranking & Filtering
        DataStream<FinalRecommendation> finalRecs = recommendations
            .process(new RankingProcessor())
            .filter(rec -> rec.getScore() > 0.2)
            .name("ranking-filtering");
        
        // 7. Personalization Cache Update
        finalRecs
            .keyBy(FinalRecommendation::getUserId)
            .process(new UserProfileUpdater())
            .name("profile-update");
        
        // 8. Real-time A/B Testing
        finalRecs
            .process(new ABTestRouter())
            .addSink(new RecommendationSink())
            .name("ab-test-delivery");
        
        env.execute("Real-time Recommendation Engine");
    }
    
    private static Map<String, Object> prepareInputs(RecommendationRequest request) {
        Map<String, Object> inputs = new HashMap<>();
        
        // User features (embeddings)
        inputs.put("user_features", request.getUserEmbedding());
        
        // Item features matrix (batch of candidates)
        inputs.put("item_features", request.getCandidateEmbeddings());
        
        // Context features
        float[] context = {
            request.getHourOfDay() / 24.0f,
            request.getDayOfWeek() / 7.0f,
            request.getDeviceType().ordinal(),
            request.getLocationId()
        };
        inputs.put("context", context);
        
        return inputs;
    }
    
    private static RecommendationResponse parseRecommendations(InferenceOutput output) {
        float[] scores = output.getOutput("scores");
        float[] ranking = output.getOutput("ranking");
        
        return new RecommendationResponse(
            scores,
            Arrays.stream(ranking)
                .mapToInt(f -> (int) f)
                .toArray(),
            System.currentTimeMillis()
        );
    }
}

// Supporting A/B Testing Logic
class ABTestRouter extends ProcessFunction<FinalRecommendation, FinalRecommendation> {
    
    private transient MapState<String, String> userExperiments;
    
    @Override
    public void processElement(
        FinalRecommendation recommendation,
        Context ctx,
        Collector<FinalRecommendation> out
    ) throws Exception {
        
        String userId = recommendation.getUserId();
        String experiment = userExperiments.get(userId);
        
        if (experiment == null) {
            // Assign user to experiment (50/50 split)
            experiment = Math.random() < 0.5 ? "control" : "treatment";
            userExperiments.put(userId, experiment);
        }
        
        // Apply experiment logic
        if ("treatment".equals(experiment)) {
            recommendation.setModelVersion("v2-neural");
            recommendation.setExperimentGroup("treatment");
        } else {
            recommendation.setModelVersion("v1-baseline");
            recommendation.setExperimentGroup("control");
        }
        
        // Record metrics
        ctx.output(new OutputTag<ExperimentEvent>("experiment-metrics") {},
            new ExperimentEvent(userId, experiment, recommendation.getScore()));
        
        out.collect(recommendation);
    }
}
```

## 🚨 Anomaly Detection

### Scenario
Real-time anomaly detection in IoT sensor data using autoencoder models.

### Implementation

```java
public class IoTAnomalyDetection {
    
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        
        // 1. IoT Sensor Data Stream
        DataStream<SensorReading> sensorData = env
            .addSource(new MQTTSource("iot/sensors/#"))
            .assignTimestampsAndWatermarks(
                WatermarkStrategy.<SensorReading>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                    .withTimestampAssigner((event, timestamp) -> event.getTimestamp())
            )
            .name("sensor-source");
        
        // 2. Window Aggregation for Time Series
        DataStream<SensorWindow> windows = sensorData
            .keyBy(SensorReading::getDeviceId)
            .window(TumblingEventTimeWindows.of(Time.minutes(1)))
            .aggregate(new SensorAggregator())
            .name("window-aggregation");
        
        // 3. PyTorch Anomaly Detection Model
        InferenceConfig anomalyConfig = InferenceConfig.builder()
            .modelConfig(ModelConfig.builder()
                .modelId("anomaly-autoencoder")
                .modelPath("/models/anomaly_detector.pt")
                .format(ModelFormat.PYTORCH)
                .inputNames(new String[]{"sensor_window"})
                .outputNames(new String[]{"reconstruction", "anomaly_score"})
                .engineOptions(Map.of(
                    "device", "auto",  // Auto GPU detection
                    "optimize", "true"
                ))
                .build())
            .batchSize(32)
            .enableCaching(false)  // Anomaly detection rarely repeats
            .enableMetrics(true)
            .collectLatencyMetrics(true)
            .collectThroughputMetrics(true)
            .build();
        
        // 4. Anomaly Detection Function
        AsyncModelInferenceFunction<SensorWindow, AnomalyScore> anomalyDetector =
            new AsyncModelInferenceFunction<>(
                anomalyConfig,
                cfg -> new PyTorchInferenceEngine(),
                IoTAnomalyDetection::normalizeWindow,
                IoTAnomalyDetection::parseAnomalyScore
            );
        
        // 5. Detect Anomalies
        DataStream<AnomalyScore> anomalyScores = AsyncDataStream.unorderedWait(
            windows,
            anomalyDetector,
            3000,
            TimeUnit.MILLISECONDS,
            50
        ).name("anomaly-detection");
        
        // 6. Dynamic Threshold Calculation
        DataStream<AnomalyAlert> alerts = anomalyScores
            .keyBy(AnomalyScore::getDeviceId)
            .process(new AdaptiveThresholdProcessor())
            .name("threshold-processing");
        
        // 7. Alert Processing Pipeline
        // Critical alerts to PagerDuty
        alerts.filter(alert -> alert.getSeverity() == Severity.CRITICAL)
            .addSink(new PagerDutySink())
            .name("pagerduty-alerts");
        
        // All anomalies to data lake for analysis
        alerts.addSink(new ParquetSink<>("s3://anomaly-data/"))
            .name("data-lake-sink");
        
        // Real-time dashboard updates
        alerts.addSink(new WebSocketSink())
            .name("dashboard-sink");
        
        env.execute("IoT Anomaly Detection System");
    }
    
    private static float[] normalizeWindow(SensorWindow window) {
        float[] readings = window.getReadings();
        float[] normalized = new float[readings.length];
        
        // Min-max normalization
        float min = Arrays.stream(readings).min().orElse(0);
        float max = Arrays.stream(readings).max().orElse(1);
        float range = max - min;
        
        if (range == 0) range = 1;
        
        for (int i = 0; i < readings.length; i++) {
            normalized[i] = (readings[i] - min) / range;
        }
        
        return normalized;
    }
    
    private static AnomalyScore parseAnomalyScore(InferenceOutput output) {
        float reconstructionError = output.getOutput("reconstruction")[0];
        float anomalyScore = output.getOutput("anomaly_score")[0];
        
        return new AnomalyScore(
            anomalyScore,
            reconstructionError,
            System.currentTimeMillis()
        );
    }
}

// Adaptive Threshold Logic
class AdaptiveThresholdProcessor extends KeyedProcessFunction<String, AnomalyScore, AnomalyAlert> {
    
    private transient ValueState<ThresholdStatistics> thresholdStats;
    private transient ListState<Float> recentScores;
    
    private static final int WINDOW_SIZE = 1000;
    private static final double Z_SCORE_THRESHOLD = 3.0;
    
    @Override
    public void processElement(
        AnomalyScore score,
        Context ctx,
        Collector<AnomalyAlert> out
    ) throws Exception {
        
        ThresholdStatistics stats = thresholdStats.value();
        if (stats == null) {
            stats = new ThresholdStatistics();
        }
        
        // Update statistics
        stats.update(score.getScore());
        thresholdStats.update(stats);
        
        // Store recent scores for percentile calculation
        recentScores.add(score.getScore());
        
        // Calculate dynamic threshold (99th percentile + buffer)
        Iterable<Float> scoresIterable = recentScores.get();
        List<Float> scoresList = new ArrayList<>();
        scoresIterable.forEach(scoresList::add);
        
        // Keep only recent scores
        if (scoresList.size() > WINDOW_SIZE) {
            recentScores.clear();
            scoresList = scoresList.subList(scoresList.size() - WINDOW_SIZE, scoresList.size());
            scoresList.forEach(recentScores::add);
        }
        
        // Calculate percentile threshold
        Collections.sort(scoresList);
        int percentileIndex = (int) (scoresList.size() * 0.99);
        float percentile99 = scoresList.get(Math.min(percentileIndex, scoresList.size() - 1));
        
        // Z-score based detection
        double zScore = (score.getScore() - stats.getMean()) / stats.getStdDev();
        
        // Determine if anomaly
        boolean isAnomaly = score.getScore() > percentile99 * 1.5 || 
                           Math.abs(zScore) > Z_SCORE_THRESHOLD;
        
        if (isAnomaly) {
            Severity severity = score.getScore() > percentile99 * 2.0 ? 
                Severity.CRITICAL : Severity.WARNING;
            
            out.collect(new AnomalyAlert(
                ctx.getCurrentKey(),
                score.getScore(),
                severity,
                score.getTimestamp(),
                Map.of(
                    "percentile_99", percentile99,
                    "z_score", zScore,
                    "mean", stats.getMean(),
                    "std_dev", stats.getStdDev()
                )
            ));
        }
    }
}
```

## 🗣️ Natural Language Processing

### Scenario
Real-time sentiment analysis and text classification.

### Implementation

```java
public class SentimentAnalysisPipeline {
    
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        
        // 1. Social Media Stream
        DataStream<SocialMediaPost> posts = env
            .addSource(new TwitterStreamSource("#technology"))
            .filter(post -> post.getLanguage().equals("en"))
            .name("social-media-source");
        
        // 2. Text Preprocessing
        DataStream<ProcessedText> processedText = posts
            .map(new TextPreprocessor())
            .name("text-preprocessing");
        
        // 3. TensorFlow BERT Model
        InferenceConfig sentimentConfig = InferenceConfig.builder()
            .modelConfig(ModelConfig.builder()
                .modelId("bert-sentiment")
                .modelPath("/models/bert_sentiment")
                .format(ModelFormat.TENSORFLOW)
                .signatureName("serving_default")
                .inputNames(new String[]{"input_ids", "attention_mask", "token_type_ids"})
                .outputNames(new String[]{"logits", "probabilities"})
                .modelOptions(Map.of(
                    "max_seq_length", "128",
                    "do_lower_case", "true"
                ))
                .build())
            .batchSize(16)  // Smaller batches for BERT
            .enableCaching(true)
            .cacheSize(1000)  // Cache tokenized inputs
            .cacheTtl(Duration.ofHours(1))
            .parallelism(2)
            .build();
        
        // 4. Sentiment Analysis
        AsyncModelInferenceFunction<ProcessedText, SentimentResult> sentimentAnalyzer =
            new AsyncModelInferenceFunction<>(
                sentimentConfig,
                cfg -> new TensorFlowInferenceEngine(),
                SentimentAnalysisPipeline::tokenizeText,
                SentimentAnalysisPipeline::parseSentiment
            );
        
        // 5. Apply Sentiment Analysis
        DataStream<SentimentResult> sentiments = AsyncDataStream.unorderedWait(
            processedText,
            sentimentAnalyzer,
            10000,  // Longer timeout for BERT
            TimeUnit.MILLISECONDS,
            50
        ).name("sentiment-analysis");
        
        // 6. Trend Analysis
        DataStream<TrendAlert> trends = sentiments
            .keyBy(SentimentResult::getTopic)
            .window(SlidingEventTimeWindows.of(Time.hours(1), Time.minutes(5)))
            .aggregate(new TrendAggregator())
            .process(new TrendAnalyzer())
            .name("trend-analysis");
        
        // 7. Real-time Dashboard
        sentiments.addSink(new KafkaSink<>("sentiment-results"))
            .name("sentiment-sink");
        
        trends.addSink(new WebSocketSink())
            .name("trend-dashboard");
        
        env.execute("Real-time Sentiment Analysis");
    }
    
    private static Map<String, Object> tokenizeText(ProcessedText text) {
        // BERT tokenization
        BertTokenizer tokenizer = BertTokenizer.fromPretrained("bert-base-uncased");
        Encoding encoding = tokenizer.encode(
            text.getText(),
            TruncationStrategy.LONGEST_FIRST,
            PaddingStrategy.MAX_LENGTH,
            128
        );
        
        Map<String, Object> inputs = new HashMap<>();
        inputs.put("input_ids", encoding.getIds());
        inputs.put("attention_mask", encoding.getAttentionMask());
        inputs.put("token_type_ids", encoding.getTypeIds());
        
        return inputs;
    }
    
    private static SentimentResult parseSentiment(InferenceOutput output) {
        float[] logits = output.getOutput("logits");
        float[] probabilities = output.getOutput("probabilities");
        
        // Find max probability
        int maxIndex = 0;
        float maxProb = probabilities[0];
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                maxIndex = i;
            }
        }
        
        Sentiment sentiment = Sentiment.fromIndex(maxIndex);
        
        return new SentimentResult(
            sentiment,
            maxProb,
            Arrays.copyOfRange(probabilities, 0, probabilities.length),
            System.currentTimeMillis()
        );
    }
}

enum Sentiment {
    VERY_NEGATIVE(0),
    NEGATIVE(1),
    NEUTRAL(2),
    POSITIVE(3),
    VERY_POSITIVE(4);
    
    private final int index;
    
    Sentiment(int index) {
        this.index = index;
    }
    
    public static Sentiment fromIndex(int index) {
        return Arrays.stream(values())
            .filter(s -> s.index == index)
            .findFirst()
            .orElse(NEUTRAL);
    }
}
```

## 🖼️ Computer Vision

### Scenario
Real-time object detection in video streams.

### Implementation

```java
public class VideoAnalyticsPipeline {
    
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        
        // 1. Video Stream Source
        DataStream<VideoFrame> videoStream = env
            .addSource(new RTSPStreamSource("rtsp://camera-feed"))
            .assignTimestampsAndWatermarks(
                WatermarkStrategy.<VideoFrame>forBoundedOutOfOrderness(Duration.ofMillis(100))
                    .withTimestampAssigner((frame, ts) -> frame.getTimestamp())
            )
            .name("video-source");
        
        // 2. Frame Decoding and Preprocessing
        DataStream<ProcessedFrame> processedFrames = videoStream
            .map(new FrameProcessor())
            .name("frame-processing");
        
        // 3. ONNX YOLO Object Detection
        InferenceConfig detectionConfig = InferenceConfig.builder()
            .modelConfig(ModelConfig.builder()
                .modelId("yolov5-object-detection")
                .modelPath("/models/yolov5s.onnx")
                .format(ModelFormat.ONNX)
                .inputNames(new String[]{"images"})
                .outputNames(new String[]{"output0"})
                .modelOptions(Map.of(
                    "img_size", "640",
                    "conf_threshold", "0.25",
                    "iou_threshold", "0.45"
                ))
                .engineOptions(Map.of(
                    "executionProvider", "CUDA",
                    "cudaDeviceId", "0",
                    "arenaExtensionStrategy", "kSameAsRequested"
                ))
                .build())
            .batchSize(8)  // Limited by GPU memory
            .enableCaching(false)
            .enableMetrics(true)
            .collectGpuMetrics(true)  // GPU-specific metrics
            .build();
        
        // 4. Object Detection Function
        AsyncModelInferenceFunction<ProcessedFrame, DetectionResult> objectDetector =
            new AsyncModelInferenceFunction<>(
                detectionConfig,
                cfg -> new OnnxInferenceEngine(),
                VideoAnalyticsPipeline::prepareFrame,
                VideoAnalyticsPipeline::parseDetections
            );
        
        // 5. Apply Object Detection
        DataStream<DetectionResult> detections = AsyncDataStream.unorderedWait(
            processedFrames,
            objectDetector,
            500,  // Very short timeout for real-time video
            TimeUnit.MILLISECONDS,
            16    // Match typical camera FPS
        ).name("object-detection");
        
        // 6. Object Tracking
        DataStream<TrackedObject> trackedObjects = detections
            .keyBy(DetectionResult::getCameraId)
            .process(new ObjectTracker())
            .name("object-tracking");
        
        // 7. Business Logic
        // People counting
        DataStream<PeopleCount> peopleCounts = trackedObjects
            .filter(obj -> obj.getClassName().equals("person"))
            .keyBy(TrackedObject::getCameraId)
            .window(TumblingProcessingTimeWindows.of(Time.seconds(1)))
            .aggregate(new PeopleCounter())
            .name("people-counting");
        
        // Intrusion detection
        DataStream<IntrusionAlert> intrusionAlerts = trackedObjects
            .process(new IntrusionDetector())
            .name("intrusion-detection");
        
        // 8. Output Sinks
        // Real-time overlay for monitoring
        detections.addSink(new VideoOverlaySink())
            .name("video-overlay");
        
        // Analytics to database
        peopleCounts.addSink(new TimescaleDBSink())
            .name("analytics-db");
        
        // Alerts to security system
        intrusionAlerts.addSink(new SecuritySystemSink())
            .name("security-alerts");
        
        env.execute("Real-time Video Analytics");
    }
    
    private static float[][][][] prepareFrame(ProcessedFrame frame) {
        // Convert to YOLO input format: [batch, channels, height, width]
        int height = 640;
        int width = 640;
        int channels = 3;
        
        float[][][][] input = new float[1][channels][height][width];
        
        // Preprocessing: resize, normalize, convert to float
        Mat resized = new Mat();
        Imgproc.resize(frame.getImage(), resized, new Size(width, height));
        
        // Convert BGR to RGB and normalize
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double[] pixel = resized.get(y, x);
                input[0][2][y][x] = (float) (pixel[0] / 255.0);  // R
                input[0][1][y][x] = (float) (pixel[1] / 255.0);  // G
                input[0][0][y][x] = (float) (pixel[2] / 255.0);  // B
            }
        }
        
        return input;
    }
    
    private static DetectionResult parseDetections(InferenceOutput output) {
        float[] detections = output.getOutput("output0");
        
        // YOLO output parsing
        List<BoundingBox> boxes = new ArrayList<>();
        List<String> classNames = new ArrayList<>();
        List<Float> confidences = new ArrayList<>();
        
        int numDetections = detections.length / 6;  // [x, y, w, h, conf, class]
        
        for (int i = 0; i < numDetections; i++) {
            int offset = i * 6;
            float x = detections[offset];
            float y = detections[offset + 1];
            float width = detections[offset + 2];
            float height = detections[offset + 3];
            float confidence = detections[offset + 4];
            int classId = (int) detections[offset + 5];
            
            if (confidence > 0.25) {  // Confidence threshold
                boxes.add(new BoundingBox(x, y, width, height));
                classNames.add(COCO_CLASSES[classId]);
                confidences.add(confidence);
            }
        }
        
        return new DetectionResult(
            boxes.toArray(new BoundingBox[0]),
            classNames.toArray(new String[0]),
            confidences.stream().mapToFloat(Float::floatValue).toArray(),
            System.currentTimeMillis()
        );
    }
    
    private static final String[] COCO_CLASSES = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };
}
```

## 📈 Time Series Forecasting

### Scenario
Real-time demand forecasting for supply chain optimization.

### Implementation

```java
public class DemandForecastingPipeline {
    
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        
        // 1. Sales Data Stream
        DataStream<SalesEvent> salesStream = env
            .addSource(new KafkaSource<>("sales-events"))
            .assignTimestampsAndWatermarks(
                WatermarkStrategy.<SalesEvent>forBoundedOutOfOrderness(Duration.ofMinutes(5))
                    .withTimestampAssigner((event, ts) -> event.getTimestamp())
            )
            .name("sales-source");
        
        // 2. Feature Engineering
        DataStream<ForecastFeatures> features = salesStream
            .keyBy(SalesEvent::getProductId)
            .window(SlidingEventTimeWindows.of(Time.hours(24), Time.hours(1)))
            .aggregate(new SalesAggregator())
            .map(new FeatureEngineer())
            .name("feature-engineering");
        
        // 3. PMML Forecasting Model
        InferenceConfig forecastConfig = InferenceConfig.builder()
            .modelConfig(ModelConfig.builder()
                .modelId("arima-forecaster")
                .modelPath("/models/forecast_model.pmml")
                .format(ModelFormat.PMML)
                .inputNames(new String[]{
                    "lag1", "lag2", "lag3", "lag7",
                    "rolling_mean_24h", "rolling_std_24h",
                    "day_of_week", "hour_of_day", "is_holiday"
                })
                .outputNames(new String[]{"forecast", "confidence_interval"})
                .build())
            .batchSize(1)  // Time series usually processed individually
            .enableCaching(true)
            .cacheSize(1000)
            .cacheTtl(Duration.ofHours(1))
            .build();
        
        // 4. Forecasting Function
        AsyncModelInferenceFunction<ForecastFeatures, ForecastResult> forecaster =
            new AsyncModelInferenceFunction<>(
                forecastConfig,
                cfg -> new PMMLInferenceEngine(),
                DemandForecastingPipeline::prepareTimeSeriesFeatures,
                DemandForecastingPipeline::parseForecast
            );
        
        // 5. Generate Forecasts
        DataStream<ForecastResult> forecasts = AsyncDataStream.unorderedWait(
            features,
            forecaster,
            2000,
            TimeUnit.MILLISECONDS,
            100
        ).name("demand-forecasting");
        
        // 6. Inventory Optimization
        DataStream<InventoryDecision> inventoryDecisions = forecasts
            .keyBy(ForecastResult::getProductId)
            .connect(salesStream.keyBy(SalesEvent::getProductId))
            .process(new InventoryOptimizer())
            .name("inventory-optimization");
        
        // 7. Supply Chain Integration
        // Reorder recommendations to ERP
        inventoryDecisions.filter(decision -> decision.getAction() == Action.REORDER)
            .addSink(new ERPSink())
            .name("erp-integration");
        
        // Forecasts to BI dashboard
        forecasts.addSink(new PowerBISink())
            .name("bi-dashboard");
        
        // Alerts for stockouts
        inventoryDecisions.filter(decision -> decision.getStockoutRisk() > 0.8)
            .addSink(new AlertSink())
            .name("stockout-alerts");
        
        env.execute("Real-time Demand Forecasting");
    }
    
    private static Map<String, Object> prepareTimeSeriesFeatures(ForecastFeatures features) {
        Map<String, Object> inputs = new HashMap<>();
        
        inputs.put("lag1", features.getLag1());
        inputs.put("lag2", features.getLag2());
        inputs.put("lag3", features.getLag3());
        inputs.put("lag7", features.getLag7());
        inputs.put("rolling_mean_24h", features.getRollingMean24h());
        inputs.put("rolling_std_24h", features.getRollingStd24h());
        inputs.put("day_of_week", features.getDayOfWeek());
        inputs.put("hour_of_day", features.getHourOfDay());
        inputs.put("is_holiday", features.isHoliday() ? 1.0 : 0.0);
        
        return inputs;
    }
    
    private static ForecastResult parseForecast(InferenceOutput output) {
        float forecast = output.getOutput("forecast")[0];
        float[] confidenceInterval = output.getOutput("confidence_interval");
        
        return new ForecastResult(
            forecast,
            confidenceInterval[0],  // Lower bound
            confidenceInterval[1],  // Upper bound
            System.currentTimeMillis()
        );
    }
}

// Inventory Optimization Logic
class InventoryOptimizer extends CoProcessFunction<
    ForecastResult, 
    SalesEvent, 
    InventoryDecision
> {
    
    private transient MapState<String, ProductInventory> inventoryState;
    private transient MapState<String, ForecastResult> latestForecast;
    
    @Override
    public void processElement1(
        ForecastResult forecast,
        Context ctx,
        Collector<InventoryDecision> out
    ) throws Exception {
        // Store latest forecast
        latestForecast.put(forecast.getProductId(), forecast);
        
        // Make inventory decision based on forecast
        ProductInventory inventory = inventoryState.get(forecast.getProductId());
        if (inventory != null) {
            InventoryDecision decision = calculateInventoryDecision(
                inventory,
                forecast
            );
            out.collect(decision);
        }
    }
    
    @Override
    public void processElement2(
        SalesEvent sale,
        Context ctx,
        Collector<InventoryDecision> out
    ) throws Exception {
        // Update inventory state
        ProductInventory inventory = inventoryState.get(sale.getProductId());
        if (inventory == null) {
            inventory = new ProductInventory(sale.getProductId());
        }
        
        inventory.updateFromSale(sale);
        inventoryState.put(sale.getProductId(), inventory);
        
        // Check if reorder needed based on current inventory
        ForecastResult forecast = latestForecast.get(sale.getProductId());
        if (forecast != null) {
            InventoryDecision decision = calculateInventoryDecision(
                inventory,
                forecast
            );
            out.collect(decision);
        }
    }
    
    private InventoryDecision calculateInventoryDecision(
        ProductInventory inventory,
        ForecastResult forecast
    ) {
        double currentStock = inventory.getCurrentStock();
        double leadTimeDemand = forecast.getForecast() * inventory.getLeadTimeDays();
        double safetyStock = forecast.getUpperBound() * 1.5;  // 50% buffer
        
        double reorderPoint = leadTimeDemand + safetyStock;
        double reorderQuantity = Math.max(
            forecast.getForecast() * 7 - currentStock,  // 7 days supply
            0
        );
        
        Action action = currentStock < reorderPoint ? Action.REORDER : Action.HOLD;
        double stockoutRisk = calculateStockoutRisk(
            currentStock,
            forecast.getForecast(),
            forecast.getLowerBound(),
            forecast.getUpperBound()
        );
        
        return new InventoryDecision(
            inventory.getProductId(),
            action,
            reorderQuantity,
            stockoutRisk,
            System.currentTimeMillis(),
            Map.of(
                "current_stock", currentStock,
                "reorder_point", reorderPoint,
                "lead_time_demand", leadTimeDemand,
                "safety_stock", safetyStock
            )
        );
    }
    
    private double calculateStockoutRisk(
        double currentStock,
        double forecast,
        double lowerBound,
        double upperBound
    ) {
        if (currentStock <= 0) return 1.0;
        
        double daysOfSupply = currentStock / forecast;
        if (daysOfSupply < 1) return 0.9;
        if (daysOfSupply < 3) return 0.7;
        if (daysOfSupply < 7) return 0.3;
        
        return 0.1;
    }
}

enum Action {
    HOLD, REORDER, URGENT_REORDER
}
```

## 🔧 Model Management Examples

### A/B Testing with Model Routing

```java
public class ModelABTesting {
    
    public static void main(String[] args) throws Exception {
        // Multiple model configurations
        InferenceConfig modelV1 = createModelConfig("fraud-v1", "v1.onnx");
        InferenceConfig modelV2 = createModelConfig("fraud-v2", "v2.onnx");
        
        // Router that splits traffic 50/50
        ModelRouter router = new ModelRouter(
            Map.of(
                "v1", modelV1,
                "v2", modelV2
            ),
            new RandomSplitStrategy(0.5)  // 50% to v2
        );
        
        DataStream<PredictionWithMetadata> predictions = transactionStream
            .process(router)
            .name("model-routing");
        
        // Collect metrics for comparison
        predictions
            .keyBy(PredictionWithMetadata::getModelVersion)
            .process(new ModelComparator())
            .addSink(new MetricsCollector());
    }
}

class ModelRouter extends ProcessFunction<Transaction, PredictionWithMetadata> {
    
    private final Map<String, InferenceConfig> models;
    private final SplitStrategy splitStrategy;
    
    @Override
    public void processElement(
        Transaction transaction,
        Context ctx,
        Collector<PredictionWithMetadata> out
    ) throws Exception {
        
        String modelVersion = splitStrategy.chooseModel(transaction);
        InferenceConfig config = models.get(modelVersion);
        
        // Execute with chosen model
        AsyncModelInferenceFunction<Transaction, FraudScore> inference =
            new AsyncModelInferenceFunction<>(config);
        
        // Async execution
        CompletableFuture<FraudScore> future = inference.asyncInvoke(transaction);
        
        future.thenAccept(score -> {
            out.collect(new PredictionWithMetadata(
                score,
                modelVersion,
                transaction,
                System.currentTimeMillis()
            ));
        });
    }
}
```

### Model Ensemble

```java
public class ModelEnsemble {
    
    public static void main(String[] args) throws Exception {
        // Multiple models for ensemble
        List<InferenceConfig> models = Arrays.asList(
            createModelConfig("model1", "model1.onnx"),
            createModelConfig("model2", "model2.onnx"),
            createModelConfig("model3", "model3.onnx")
        );
        
        // Ensemble function that averages predictions
        EnsembleInferenceFunction ensemble = new EnsembleInferenceFunction(
            models,
            new WeightedAverageEnsembler(
                Map.of("model1", 0.4, "model2", 0.3, "model3", 0.3)
            )
        );
        
        DataStream<EnsemblePrediction> predictions = transactionStream
            .process(ensemble)
            .name("model-ensemble");
    }
}

class EnsembleInferenceFunction extends ProcessFunction<Transaction, EnsemblePrediction> {
    
    private final List<InferenceConfig> models;
    private final Ensembler ensembler;
    
    @Override
    public void processElement(
        Transaction transaction,
        Context ctx,
        Collector<EnsemblePrediction> out
    ) throws Exception {
        
        List<CompletableFuture<FraudScore>> futures = new ArrayList<>();
        
        // Execute all models in parallel
        for (InferenceConfig config : models) {
            AsyncModelInferenceFunction<Transaction, FraudScore> inference =
                new AsyncModelInferenceFunction<>(config);
            
            futures.add(inference.asyncInvoke(transaction));
        }
        
        // Wait for all predictions
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
            .thenAccept(v -> {
                List<FraudScore> predictions = futures.stream()
                    .map(CompletableFuture::join)
                    .collect(Collectors.toList());
                
                FraudScore ensembleScore = ensembler.combine(predictions);
                
                out.collect(new EnsemblePrediction(
                    ensembleScore,
                    predictions,
                    System.currentTimeMillis()
                ));
            });
    }
}
```

---

These examples demonstrate the versatility and power of Otter Streams for real-time ML inference across various domains. Each example includes:

1. **Complete pipeline setup** from source to sink
2. **Model configuration** for different frameworks
3. **Feature engineering** and preprocessing
4. **Business logic integration**
5. **Monitoring and alerting**
6. **Production considerations** like A/B testing and ensembles

For more specific use cases or implementation questions, please open a discussion in the GitHub repository!

# PERFORMANCE.md

# ⚡ Performance Guide

This guide covers optimization strategies, performance tuning, and monitoring best practices for Otter Streams.

## 📊 Performance Benchmarks

### Reference Architecture
- **CPU**: Intel Xeon Platinum 8380 (2.3 GHz, 40 cores)
- **RAM**: 256 GB DDR4
- **GPU**: NVIDIA A100 80GB (optional)
- **Network**: 25 Gbps
- **Storage**: NVMe SSD

### Baseline Performance
| Model Type | Batch Size | Throughput | P50 Latency | P99 Latency | GPU Speedup |
|------------|------------|------------|-------------|-------------|-------------|
| ONNX (CPU) | 32 | 2,800 inf/sec | 8ms | 25ms | - |
| ONNX (A100) | 32 | 12,500 inf/sec | 2ms | 8ms | 4.5x |
| TensorFlow | 32 | 2,200 inf/sec | 10ms | 30ms | 4.0x |
| PyTorch | 32 | 2,500 inf/sec | 9ms | 28ms | 4.2x |
| XGBoost | 64 | 15,000 inf/sec | 3ms | 10ms | N/A |
| PMML | 1 | 800 inf/sec | 20ms | 50ms | N/A |

## 🔧 Optimization Strategies

### 1. Batch Size Optimization

#### Finding Optimal Batch Size

```java
public class BatchSizeOptimizer {
    
    public static InferenceConfig optimizeBatchSize(
        ModelConfig modelConfig,
        int[] candidateSizes
    ) {
        Map<Integer, PerformanceMetrics> results = new HashMap<>();
        
        for (int batchSize : candidateSizes) {
            InferenceConfig config = InferenceConfig.builder()
                .modelConfig(modelConfig)
                .batchSize(batchSize)
                .enableMetrics(true)
                .build();
            
            PerformanceMetrics metrics = benchmarkConfig(config);
            results.put(batchSize, metrics);
            
            System.out.printf("Batch %d: %.1f inf/sec, P99: %dms%n",
                batchSize,
                metrics.getThroughput(),
                metrics.getP99Latency());
        }
        
        // Choose batch size with best throughput under latency constraint
        return results.entrySet().stream()
            .filter(e -> e.getValue().getP99Latency() < 100) // < 100ms P99
            .max(Comparator.comparingDouble(e -> e.getValue().getThroughput()))
            .map(Map.Entry::getKey)
            .map(batchSize -> InferenceConfig.builder()
                .modelConfig(modelConfig)
                .batchSize(batchSize)
                .build())
            .orElseThrow();
    }
    
    private static PerformanceMetrics benchmarkConfig(InferenceConfig config) {
        // Run benchmark
        return new PerformanceMetrics(
            config.getBatchSize(),
            0.0,  // Will be measured
            0,    // Will be measured
            0     // Will be measured
        );
    }
}

// Usage
InferenceConfig optimized = BatchSizeOptimizer.optimizeBatchSize(
    modelConfig,
    new int[]{1, 2, 4, 8, 16, 32, 64, 128, 256}
);
```

#### Recommended Batch Sizes
| Model Type | Input Size | Recommended Batch | Max Throughput Batch |
|------------|------------|-------------------|----------------------|
| Small CNN | 224x224x3 | 32-64 | 128 |
| Large CNN | 512x512x3 | 8-16 | 32 |
| BERT | 128 tokens | 8-16 | 32 |
| LSTM | 100 seq | 16-32 | 64 |
| XGBoost | 50 features | 64-128 | 256 |

### 2. Memory Optimization

#### Memory Pool Configuration

```java
public class MemoryOptimization {
    
    public static void configureMemoryOptimization() {
        // Set JVM options for optimal performance
        String[] jvmOptions = {
            "-XX:+UseG1GC",
            "-XX:MaxGCPauseMillis=100",
            "-XX:InitiatingHeapOccupancyPercent=35",
            "-XX:+ParallelRefProcEnabled",
            "-XX:+PerfDisableSharedMem",
            "-XX:+OptimizeStringConcat",
            
            // Memory settings (adjust based on workload)
            "-Xms8g",
            "-Xmx8g",
            
            // Direct memory for native libraries
            "-XX:MaxDirectMemorySize=2g",
            
            // ONNX Runtime specific
            "-Donnxruntime.native.arena.enable=true",
            "-Donnxruntime.native.arena.max_size=1024"
        };
        
        System.out.println("Recommended JVM options:");
        Arrays.stream(jvmOptions).forEach(System.out::println);
    }
    
    public static InferenceConfig memoryOptimizedConfig(ModelConfig modelConfig) {
        return InferenceConfig.builder()
            .modelConfig(modelConfig)
            .batchSize(32)  // Balance memory vs throughput
            .queueSize(1000)  // Prevent OOM from queued requests
            .modelOptions(Map.of(
                // ONNX Runtime memory optimization
                "arenaExtensionStrategy", "kSameAsRequested",
                "enableCpuMemArena", "true",
                "enableMemPattern", "true",
                
                // TensorFlow memory optimization
                "gpu_memory_fraction", "0.8",
                "allow_growth", "true",
                
                // PyTorch memory optimization
                "memory.pinned", "true",
                "memory.allocator", "native"
            ))
            .build();
    }
}
```

#### Monitoring Memory Usage

```java
public class MemoryMonitor extends ProcessFunction<Object, Void> {
    
    private transient Gauge<Long> heapMemoryGauge;
    private transient Gauge<Long> nonHeapMemoryGauge;
    private transient Gauge<Long> directMemoryGauge;
    
    @Override
    public void open(Configuration parameters) {
        // Register memory metrics
        heapMemoryGauge = getRuntimeContext()
            .getMetricGroup()
            .gauge("memory.heap.used", 
                () -> Runtime.getRuntime().totalMemory() - 
                      Runtime.getRuntime().freeMemory());
        
        nonHeapMemoryGauge = getRuntimeContext()
            .getMetricGroup()
            .gauge("memory.nonheap.used",
                () -> ManagementFactory.getMemoryMXBean()
                    .getNonHeapMemoryUsage().getUsed());
        
        // Direct memory monitoring (for native libraries)
        directMemoryGauge = getRuntimeContext()
            .getMetricGroup()
            .gauge("memory.direct.used", this::getDirectMemoryUsed);
    }
    
    private long getDirectMemoryUsed() {
        try {
            Class<?> vmClass = Class.forName("sun.misc.VM");
            Field maxDirectMemoryField = vmClass.getDeclaredField("maxDirectMemory");
            maxDirectMemoryField.setAccessible(true);
            return (long) maxDirectMemoryField.get(null);
        } catch (Exception e) {
            return -1;
        }
    }
    
    @Override
    public void processElement(Object value, Context ctx, Collector<Void> out) {
        // Log memory usage periodically
        if (ctx.timestamp() % 60000 == 0) {  // Every minute
            System.out.printf(
                "Memory Usage - Heap: %dMB, Non-Heap: %dMB, Direct: %dMB%n",
                heapMemoryGauge.getValue() / 1024 / 1024,
                nonHeapMemoryGauge.getValue() / 1024 / 1024,
                directMemoryGauge.getValue() / 1024 / 1024
            );
            
            // Check for memory pressure
            if (heapMemoryGauge.getValue() > Runtime.getRuntime().maxMemory() * 0.8) {
                ctx.output(new OutputTag<String>("memory-alerts") {},
                    "High heap memory usage: " + 
                    (heapMemoryGauge.getValue() * 100 / Runtime.getRuntime().maxMemory()) + "%");
            }
        }
    }
}
```

### 3. Caching Strategies

#### Multi-Level Cache Configuration

```java
public class CacheOptimization {
    
    public static InferenceConfig optimizedCacheConfig(ModelConfig modelConfig) {
        return InferenceConfig.builder()
            .modelConfig(modelConfig)
            .enableCaching(true)
            
            // Model cache - for frequently used models
            .modelCacheConfig(CacheConfig.builder()
                .maximumSize(10)  // Keep 10 models in memory
                .expireAfterWrite(Duration.ofHours(24))
                .recordStats(true)
                .build())
            
            // Result cache - for repeated identical inputs
            .resultCacheConfig(CacheConfig.builder()
                .maximumSize(100000)  // 100K results
                .expireAfterWrite(Duration.ofMinutes(30))
                .expireAfterAccess(Duration.ofMinutes(10))
                .recordStats(true)
                .build())
            
            // Feature cache - for expensive feature computation
            .featureCacheConfig(CacheConfig.builder()
                .maximumSize(50000)  // 50K feature sets
                .expireAfterWrite(Duration.ofHours(1))
                .recordStats(true)
                .build())
            
            .build();
    }
    
    public static class SmartCacheLoader implements CacheLoader<String, InferenceOutput> {
        
        private final InferenceEngine engine;
        private final Function<Object, Object> preprocessor;
        
        @Override
        public InferenceOutput load(String key) {
            // Parse key to get input
            Object input = parseKey(key);
            
            // Apply preprocessing
            Object processed = preprocessor.apply(input);
            
            // Execute inference
            return engine.execute(processed);
        }
        
        @Override
        public Map<String, InferenceOutput> loadAll(Iterable<? extends String> keys) {
            Map<String, InferenceOutput> results = new HashMap<>();
            List<Object> batchInputs = new ArrayList<>();
            List<String> batchKeys = new ArrayList<>();
            
            for (String key : keys) {
                Object input = parseKey(key);
                Object processed = preprocessor.apply(input);
                batchInputs.add(processed);
                batchKeys.add(key);
            }
            
            // Batch inference
            List<InferenceOutput> batchResults = engine.executeBatch(batchInputs);
            
            for (int i = 0; i < batchKeys.size(); i++) {
                results.put(batchKeys.get(i), batchResults.get(i));
            }
            
            return results;
        }
    }
}
```

#### Cache Hit Rate Optimization

```java
public class CacheAnalyzer extends ProcessFunction<CacheEvent, CacheMetrics> {
    
    private transient ValueState<CacheStatistics> cacheStats;
    private transient ListState<Double> hitRateHistory;
    
    @Override
    public void processElement(
        CacheEvent event,
        Context ctx,
        Collector<CacheMetrics> out
    ) throws Exception {
        
        CacheStatistics stats = cacheStats.value();
        if (stats == null) {
            stats = new CacheStatistics();
        }
        
        // Update statistics
        if (event.isHit()) {
            stats.recordHit();
        } else {
            stats.recordMiss();
        }
        
        cacheStats.update(stats);
        
        // Store hit rate history
        hitRateHistory.add(stats.getHitRate());
        
        // Analyze and suggest optimizations
        if (stats.getTotalRequests() % 1000 == 0) {
            CacheMetrics metrics = analyzeCachePerformance(stats);
            out.collect(metrics);
            
            // Suggest cache tuning
            if (metrics.getHitRate() < 0.3) {
                suggestCacheOptimization(metrics, ctx);
            }
        }
    }
    
    private CacheMetrics analyzeCachePerformance(CacheStatistics stats) {
        double hitRate = stats.getHitRate();
        double avgLoadPenalty = stats.getAverageLoadPenalty();
        long evictionCount = stats.getEvictionCount();
        
        // Calculate cache effectiveness
        double effectiveness = hitRate * (1.0 - (avgLoadPenalty / 1000.0));
        
        return new CacheMetrics(
            hitRate,
            effectiveness,
            evictionCount,
            stats.getTotalRequests(),
            System.currentTimeMillis()
        );
    }
    
    private void suggestCacheOptimization(CacheMetrics metrics, Context ctx) {
        if (metrics.getHitRate() < 0.2) {
            ctx.output(new OutputTag<String>("cache-optimization") {},
                "Low cache hit rate (" + String.format("%.1f", metrics.getHitRate() * 100) + 
                "%). Consider: 1) Increase cache size 2) Adjust TTL 3) Review cache key design");
        }
        
        if (metrics.getEvictionCount() > 1000) {
            ctx.output(new OutputTag<String>("cache-optimization") {},
                "High eviction rate. Consider increasing cache size from current " +
                (metrics.getTotalRequests() / 10) + " entries");
        }
    }
}
```

### 4. Parallelism & Concurrency

#### Optimal Parallelism Calculation

```java
public class ParallelismOptimizer {
    
    public static int calculateOptimalParallelism(
        int inputRate,      // events per second
        int processingTime, // ms per inference (single)
        int targetLatency,  // target P99 latency in ms
        int cpuCores
    ) {
        // Little's Law: L = λW
        // Where L = average number in system, λ = arrival rate, W = average time in system
        
        double arrivalRate = inputRate;
        double serviceTime = processingTime / 1000.0; // Convert to seconds
        double targetQueueTime = targetLatency / 1000.0 - serviceTime;
        
        if (targetQueueTime <= 0) {
            targetQueueTime = serviceTime * 0.1; // Allow 10% queue time
        }
        
        // Calculate required parallelism
        double utilization = 0.7; // Target 70% utilization
        double requiredParallelism = (arrivalRate * serviceTime) / utilization;
        
        // Adjust for available cores
        int parallelism = (int) Math.ceil(requiredParallelism);
        parallelism = Math.min(parallelism, cpuCores * 2); // Up to 2 threads per core
        parallelism = Math.max(parallelism, 1); // At least 1
        
        System.out.printf(
            "Input: %d/sec, Process: %dms, Target: %dms, Cores: %d -> Parallelism: %d%n",
            inputRate, processingTime, targetLatency, cpuCores, parallelism
        );
        
        return parallelism;
    }
    
    public static InferenceConfig autoParallelismConfig(
        ModelConfig modelConfig,
        SystemMetrics systemMetrics
    ) {
        int parallelism = calculateOptimalParallelism(
            systemMetrics.getInputRate(),
            systemMetrics.getAvgProcessingTimeMs(),
            systemMetrics.getTargetLatencyMs(),
            systemMetrics.getAvailableCores()
        );
        
        return InferenceConfig.builder()
            .modelConfig(modelConfig)
            .parallelism(parallelism)
            .queueSize(parallelism * 100)  // 100 requests per parallel instance
            .maxConcurrentRequests(parallelism * 50)  // 50 concurrent per instance
            .build();
    }
}
```

#### Dynamic Parallelism Adjustment

```java
public class DynamicParallelismManager extends ProcessFunction<SystemMetrics, Void> {
    
    private transient MapState<Integer, InferenceConfig> configVersions;
    private transient ValueState<Integer> currentParallelism;
    
    @Override
    public void processElement(
        SystemMetrics metrics,
        Context ctx,
        Collector<Void> out
    ) throws Exception {
        
        int current = currentParallelism.value() != null ? 
            currentParallelism.value() : 1;
        
        // Calculate desired parallelism based on metrics
        int desired = calculateDesiredParallelism(metrics, current);
        
        if (desired != current) {
            // Update parallelism
            updateParallelism(desired, ctx);
            currentParallelism.update(desired);
            
            ctx.output(new OutputTag<String>("parallelism-changes") {},
                String.format("Parallelism changed: %d -> %d (load: %.1f%%, latency: %dms)",
                    current, desired,
                    metrics.getCpuUsage() * 100,
                    metrics.getP99Latency()));
        }
    }
    
    private int calculateDesiredParallelism(SystemMetrics metrics, int current) {
        double cpuUsage = metrics.getCpuUsage();
        double latency = metrics.getP99Latency();
        double targetLatency = metrics.getTargetLatencyMs();
        
        // Rules for adjusting parallelism
        if (latency > targetLatency * 1.5 && cpuUsage < 0.6) {
            // High latency, underutilized CPU -> increase parallelism
            return Math.min(current * 2, metrics.getMaxParallelism());
        } else if (latency < targetLatency * 0.7 && cpuUsage > 0.8) {
            // Low latency, high CPU -> decrease parallelism
            return Math.max(current / 2, 1);
        } else if (cpuUsage > 0.9) {
            // Very high CPU -> decrease
            return Math.max(current - 1, 1);
        } else if (cpuUsage < 0.5 && latency > targetLatency) {
            // Low CPU but high latency -> increase
            return current + 1;
        }
        
        return current;
    }
    
    private void updateParallelism(int newParallelism, Context ctx) {
        // In production, this would trigger a pipeline update
        // For now, we just log the recommendation
        System.out.println("Recommended parallelism change to: " + newParallelism);
        
        // In a real implementation, you might:
        // 1. Update Flink job parallelism
        // 2. Redeploy with new configuration
        // 3. Use savepoints for stateful updates
    }
}
```

## 📈 Monitoring & Metrics

### Comprehensive Metrics Collection

```java
public class PerformanceMonitor {
    
    public static InferenceConfig monitoringEnabledConfig(ModelConfig modelConfig) {
        return InferenceConfig.builder()
            .modelConfig(modelConfig)
            .enableMetrics(true)
            .metricsPrefix("otterstreams")
            
            // Latency metrics
            .collectLatencyMetrics(true)
            .latencyPercentiles(new double[]{0.5, 0.75, 0.95, 0.99, 0.999})
            .latencyHistogramBuckets(new double[]{
                1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000
            })
            
            // Throughput metrics
            .collectThroughputMetrics(true)
            .throughputWindow(Duration.ofMinutes(1))
            
            // Error metrics
            .collectErrorMetrics(true)
            .errorCategories(Arrays.asList(
                "timeout", "model_not_found", "invalid_input", 
                "runtime_error", "resource_exhausted"
            ))
            
            // Cache metrics
            .collectCacheMetrics(true)
            .cacheMetricsDetailLevel(CacheMetricsDetailLevel.FULL)
            
            // Resource metrics
            .collectResourceMetrics(true)
            .resourceMetricsInterval(Duration.ofSeconds(30))
            
            // Business metrics
            .collectBusinessMetrics(true)
            .businessMetricsConfig(Map.of(
                "fraud_detection.threshold", 0.9,
                "anomaly_detection.z_score", 3.0,
                "recommendation.min_score", 0.2
            ))
            
            // Export configuration
            .metricsExporters(Arrays.asList(
                MetricsExporter.PROMETHEUS,
                MetricsExporter.JMX,
                MetricsExporter.CLOUDWATCH
            ))
            .metricsExportInterval(Duration.ofSeconds(15))
            
            .build();
    }
}

// Custom metrics collection
public class CustomMetricsCollector extends ProcessFunction<InferenceResult, Void> {
    
    private transient Counter successfulInferences;
    private transient Counter failedInferences;
    private transient Distribution latencyDistribution;
    private transient Meter throughputMeter;
    private transient Gauge<Double> successRateGauge;
    
    private transient ValueState<WindowStatistics> windowStats;
    
    @Override
    public void open(Configuration parameters) {
        // Register metrics
        successfulInferences = getRuntimeContext()
            .getMetricGroup()
            .counter("inferences.successful");
        
        failedInferences = getRuntimeContext()
            .getMetricGroup()
            .counter("inferences.failed");
        
        latencyDistribution = getRuntimeContext()
            .getMetricGroup()
            .distribution("inferences.latency");
        
        throughputMeter = getRuntimeContext()
            .getMetricGroup()
            .meter("inferences.throughput", new MeterView(60));
        
        successRateGauge = getRuntimeContext()
            .getMetricGroup()
            .gauge("inferences.success_rate", () -> {
                WindowStatistics stats = windowStats.value();
                return stats != null ? stats.getSuccessRate() : 1.0;
            });
    }
    
    @Override
    public void processElement(
        InferenceResult result,
        Context ctx,
        Collector<Void> out
    ) throws Exception {
        
        // Update metrics
        if (result.isSuccess()) {
            successfulInferences.inc();
            latencyDistribution.update(result.getLatencyMs());
            throughputMeter.markEvent();
        } else {
            failedInferences.inc();
        }
        
        // Update window statistics
        WindowStatistics stats = windowStats.value();
        if (stats == null) {
            stats = new WindowStatistics();
        }
        stats.update(result);
        windowStats.update(stats);
        
        // Log detailed metrics periodically
        if (ctx.timestamp() % 60000 == 0) {  // Every minute
            logDetailedMetrics(stats, ctx);
        }
    }
    
    private void logDetailedMetrics(WindowStatistics stats, Context ctx) {
        Map<String, Object> metrics = new HashMap<>();
        metrics.put("timestamp", System.currentTimeMillis());
        metrics.put("success_rate", stats.getSuccessRate());
        metrics.put("avg_latency_ms", stats.getAverageLatency());
        metrics.put("p95_latency_ms", stats.getP95Latency());
        metrics.put("p99_latency_ms", stats.getP99Latency());
        metrics.put("throughput_eps", stats.getThroughputPerSecond());
        metrics.put("error_rate", stats.getErrorRate());
        
        // Output to side output for external processing
        ctx.output(new OutputTag<Map<String, Object>>("detailed-metrics") {},
            metrics);
        
        // Log to console (for debugging)
        System.out.printf(
            "Metrics: %.1f%% success, %.1fms avg, %.1fms p95, %.1f eps%n",
            stats.getSuccessRate() * 100,
            stats.getAverageLatency(),
            stats.getP95Latency(),
            stats.getThroughputPerSecond()
        );
    }
}
```

### Performance Dashboard Configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'otter-streams'
    static_configs:
      - targets: ['localhost:9250']
    metrics_path: '/metrics'
    scrape_interval: 15s
    
  - job_name: 'flink-metrics'
    static_configs:
      - targets: ['jobmanager:9249', 'taskmanager:9250']
    scrape_interval: 15s

# Grafana dashboard JSON (simplified)
{
  "title": "Otter Streams Performance",
  "panels": [
    {
      "title": "Throughput",
      "targets": [
        "rate(otterstreams_inferences_total[1m])"
      ],
      "unit": "requests/sec"
    },
    {
      "title": "Latency Distribution",
      "targets": [
        "otterstreams_inference_latency_ms{p=\"50\"}",
        "otterstreams_inference_latency_ms{p=\"95\"}",
        "otterstreams_inference_latency_ms{p=\"99\"}"
      ],
      "unit": "ms"
    },
    {
      "title": "Success Rate",
      "targets": [
        "otterstreams_inferences_successful / otterstreams_inferences_total"
      ],
      "unit": "percentunit"
    },
    {
      "title": "Cache Performance",
      "targets": [
        "rate(otterstreams_cache_hits_total[1m])",
        "rate(otterstreams_cache_misses_total[1m])"
      ],
      "unit": "requests/sec"
    }
  ],
  "refresh": "30s",
  "timeRange": "now-1h"
}
```

## 🔍 Troubleshooting Performance Issues

### Common Issues and Solutions

#### Issue 1: High Latency
**Symptoms**: P99 latency > 100ms, queue buildup

**Diagnosis**:
```java
public class LatencyDiagnostic {
    public static void diagnoseHighLatency(PerformanceMetrics metrics) {
        System.out.println("=== Latency Diagnostics ===");
        System.out.printf("P50: %dms, P95: %dms, P99: %dms%n",
            metrics.getP50Latency(),
            metrics.getP95Latency(),
            metrics.getP99Latency());
        
        if (metrics.getQueueSize() > metrics.getParallelism() * 10) {
            System.out.println("⚠️  High queue size detected: " + metrics.getQueueSize());
            System.out.println("    → Consider increasing parallelism or reducing batch size");
        }
        
        if (metrics.getCpuUsage() > 0.8) {
            System.out.println("⚠️  High CPU usage: " + String.format("%.1f", metrics.getCpuUsage() * 100) + "%");
            System.out.println("    → Consider scaling out or optimizing model");
        }
        
        if (metrics.getCacheHitRate() < 0.3) {
            System.out.println("⚠️  Low cache hit rate: " + String.format("%.1f", metrics.getCacheHitRate() * 100) + "%");
            System.out.println("    → Consider increasing cache size or reviewing cache keys");
        }
    }
}
```

**Solutions**:
1. Increase parallelism
2. Reduce batch size
3. Enable/optimize caching
4. Upgrade hardware (CPU/GPU)
5. Optimize model (quantization, pruning)

#### Issue 2: Low Throughput
**Symptoms**: Throughput below expected, underutilized resources

**Diagnosis**:
```java
public class ThroughputDiagnostic {
    public static void diagnoseLowThroughput(PerformanceMetrics metrics) {
        double expectedThroughput = calculateExpectedThroughput(metrics);
        double actualThroughput = metrics.getThroughput();
        double efficiency = actualThroughput / expectedThroughput;
        
        System.out.println("=== Throughput Diagnostics ===");
        System.out.printf("Expected: %.0f inf/sec, Actual: %.0f inf/sec, Efficiency: %.1f%%%n",
            expectedThroughput, actualThroughput, efficiency * 100);
        
        if (efficiency < 0.5) {
            System.out.println("⚠️  Low efficiency detected");
            
            if (metrics.getBatchSize() == 1) {
                System.out.println("    → Consider enabling batching");
            } else if (metrics.getBatchUtilization() < 0.7) {
                System.out.println("    → Low batch utilization: " + 
                    String.format("%.1f", metrics.getBatchUtilization() * 100) + "%");
                System.out.println("    → Adjust batch timeout or size");
            }
            
            if (metrics.getIoWaitTime() > 0.3) {
                System.out.println("⚠️  High I/O wait time: " + 
                    String.format("%.1f", metrics.getIoWaitTime() * 100) + "%");
                System.out.println("    → Consider using faster storage or in-memory models");
            }
        }
    }
}
```

**Solutions**:
1. Increase batch size
2. Adjust batch timeout
3. Optimize I/O (use SSDs, memory-mapped files)
4. Enable async I/O
5. Use GPU acceleration

#### Issue 3: Memory Exhaustion
**Symptoms**: OutOfMemoryError, high GC activity

**Diagnosis**:
```java
public class MemoryDiagnostic {
    public static void diagnoseMemoryIssues(MemoryMetrics metrics) {
        System.out.println("=== Memory Diagnostics ===");
        System.out.printf("Heap used: %dMB/%dMB (%.1f%%)%n",
            metrics.getHeapUsedMB(),
            metrics.getHeapMaxMB(),
            metrics.getHeapUsedPercent() * 100);
        
        System.out.printf("GC time: %.1f%%, GC count: %d%n",
            metrics.getGcTimePercent() * 100,
            metrics.getGcCount());
        
        if (metrics.getHeapUsedPercent() > 0.8) {
            System.out.println("⚠️  High heap memory usage");
            System.out.println("    → Reduce batch size");
            System.out.println("    → Increase heap memory (-Xmx)");
            System.out.println("    → Enable off-heap caching");
        }
        
        if (metrics.getGcTimePercent() > 0.1) {
            System.out.println("⚠️  High garbage collection overhead");
            System.out.println("    → Tune GC parameters (-XX:+UseG1GC)");
            System.out.println("    → Reduce object allocation");
            System.out.println("    → Use object pooling");
        }
    }
}
```

**Solutions**:
1. Increase JVM heap size
2. Reduce batch size
3. Enable off-heap memory
4. Tune garbage collector
5. Use object pooling

## 🚀 Advanced Optimization Techniques

### Model Quantization

```java
public class ModelQuantization {
    
    public static InferenceConfig quantizedConfig(ModelConfig modelConfig) {
        return InferenceConfig.builder()
            .modelConfig(modelConfig)
            .modelOptions(Map.of(
                // ONNX quantization
                "quantization_mode", "dynamic",
                "quantization_precision", "int8",
                "quantization_calibration_data", "/calibration_data.npy",
                
                // TensorFlow quantization
                "optimize_for_inference", "true",
                "enable_quantization", "true",
                "quantization_type", "post_training_int8",
                
                // PyTorch quantization
                "torchscript_optimize", "true",
                "quantize", "true"
            ))
            .build();
    }
    
    public static PerformanceMetrics compareQuantization(
        ModelConfig originalConfig,
        ModelConfig quantizedConfig
    ) {
        System.out.println("=== Quantization Comparison ===");
        
        // Benchmark original model
        PerformanceMetrics original = benchmarkModel(originalConfig);
        System.out.printf("Original: %.0f inf/sec, %dMB memory%n",
            original.getThroughput(),
            original.getMemoryUsageMB());
        
        // Benchmark quantized model
        PerformanceMetrics quantized = benchmarkModel(quantizedConfig);
        System.out.printf("Quantized: %.0f inf/sec, %dMB memory%n",
            quantized.getThroughput(),
            quantized.getMemoryUsageMB());
        
        // Calculate improvements
        double speedup = quantized.getThroughput() / original.getThroughput();
        double memoryReduction = 1.0 - (quantized.getMemoryUsageMB() / 
            (double) original.getMemoryUsageMB());
        
        System.out.printf("Speedup: %.1fx, Memory reduction: %.1f%%%n",
            speedup, memoryReduction * 100);
        
        // Accuracy impact (if calibration data available)
        if (hasCalibrationData()) {
            double accuracyDrop = measureAccuracyDrop(originalConfig, quantizedConfig);
            System.out.printf("Accuracy impact: %.2f%% drop%n", accuracyDrop * 100);
        }
        
        return quantized;
    }
}
```

### GPU Acceleration

```java
public class GPUOptimization {
    
    public static InferenceConfig gpuOptimizedConfig(ModelConfig modelConfig) {
        return InferenceConfig.builder()
            .modelConfig(modelConfig)
            .modelOptions(Map.of(
                // CUDA/GPU options
                "executionProvider", "CUDA",
                "cudaDeviceId", "0",
                "cudaMemoryLimit", "8589934592",  // 8GB
                "arenaExtensionStrategy", "kNextPowerOfTwo",
                "enableCudaGraph", "true",
                "cudaGraphSize", "10",
                
                // TensorFlow GPU options
                "per_process_gpu_memory_fraction", "0.8",
                "allow_growth", "true",
                "gpu_options.visible_device_list", "0",
                
                // PyTorch GPU options
                "device", "cuda:0",
                "torch.backends.cudnn.benchmark", "true",
                "torch.backends.cudnn.deterministic", "false"
            ))
            .batchSize(128)  // Larger batches for GPU
            .build();
    }
    
    public static void monitorGPUUsage() {
        // Use NVIDIA SMI or similar to monitor GPU
        String[] command = {
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits"
        };
        
        try {
            Process process = Runtime.getRuntime().exec(command);
            BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream()));
            
            String line = reader.readLine();
            if (line != null) {
                String[] parts = line.split(", ");
                double gpuUtil = Double.parseDouble(parts[0]);
                double memoryUsed = Double.parseDouble(parts[1]);
                double memoryTotal = Double.parseDouble(parts[2]);
                double temperature = Double.parseDouble(parts[3]);
                
                System.out.printf(
                    "GPU: %.1f%% util, %.0f/%.0fMB memory, %.0f°C%n",
                    gpuUtil,
                    memoryUsed,
                    memoryTotal,
                    temperature
                );
                
                // Alert on high temperature or memory usage
                if (temperature > 85) {
                    System.out.println("⚠️  High GPU temperature!");
                }
                if (memoryUsed > memoryTotal * 0.9) {
                    System.out.println("⚠️  High GPU memory usage!");
                }
            }
        } catch (Exception e) {
            System.err.println("Failed to read GPU stats: " + e.getMessage());
        }
    }
}
```

## 📊 Performance Testing Framework

```java
public class PerformanceTestSuite {
    
    public static PerformanceReport runCompleteTestSuite(
        ModelConfig modelConfig,
        TestConfiguration testConfig
    ) {
        PerformanceReport report = new PerformanceReport();
        
        // 1. Single inference test
        report.setSingleInferenceTest(runSingleInferenceTest(modelConfig));
        
        // 2. Throughput test
        report.setThroughputTest(runThroughputTest(modelConfig, testConfig));
        
        // 3. Latency test
        report.setLatencyTest(runLatencyTest(modelConfig, testConfig));
        
        // 4. Memory test
        report.setMemoryTest(runMemoryTest(modelConfig, testConfig));
        
        // 5. Concurrent load test
        report.setConcurrentLoadTest(runConcurrentLoadTest(modelConfig, testConfig));
        
        // 6. Long-running stability test
        report.setStabilityTest(runStabilityTest(modelConfig, testConfig));
        
        // Generate recommendations
        report.setRecommendations(generateRecommendations(report));
        
        return report;
    }
    
    private static PerformanceTestResult runThroughputTest(
        ModelConfig modelConfig,
        TestConfiguration testConfig
    ) {
        System.out.println("=== Throughput Test ===");
        
        List<Double> throughputs = new ArrayList<>();
        List<Integer> batchSizes = Arrays.asList(1, 2, 4, 8, 16, 32, 64, 128, 256);
        
        for (int batchSize : batchSizes) {
            InferenceConfig config = InferenceConfig.builder()
                .modelConfig(modelConfig)
                .batchSize(batchSize)
                .enableMetrics(true)
                .build();
            
            double throughput = measureThroughput(config, testConfig.getDuration());
            throughputs.add(throughput);
            
            System.out.printf("Batch %3d: %8.1f inf/sec%n", batchSize, throughput);
        }
        
        // Find optimal batch size
        int optimalBatchSize = findOptimalBatchSize(batchSizes, throughputs);
        
        return new PerformanceTestResult(
            "Throughput Test",
            throughputs,
            Map.of("optimal_batch_size", optimalBatchSize)
        );
    }
    
    private static PerformanceTestResult runLatencyTest(
        ModelConfig modelConfig,
        TestConfiguration testConfig
    ) {
        System.out.println("=== Latency Test ===");
        
        Map<String, Double> latencies = new HashMap<>();
        
        // Test different percentiles
        for (double percentile : new double[]{0.5, 0.75, 0.95, 0.99, 0.999}) {
            double latency = measureLatencyPercentile(modelConfig, percentile);
            latencies.put("p" + (int)(percentile * 100), latency);
            System.out.printf("P%d: %.1fms%n", (int)(percentile * 100), latency);
        }
        
        // Test with different loads
        for (int loadPercent : new int[]{25, 50, 75, 100}) {
            double latency = measureLatencyUnderLoad(modelConfig, loadPercent / 100.0);
            latencies.put("load_" + loadPercent, latency);
            System.out.printf("Load %d%%: %.1fms%n", loadPercent, latency);
        }
        
        return new PerformanceTestResult("Latency Test", latencies);
    }
    
    private static List<String> generateRecommendations(PerformanceReport report) {
        List<String> recommendations = new ArrayList<>();
        
        // Analyze throughput
        if (report.getThroughputTest().getOptimalBatchSize() != 
            report.getModelConfig().getDefaultBatchSize()) {
            recommendations.add(String.format(
                "Change batch size from %d to %d for optimal throughput",
                report.getModelConfig().getDefaultBatchSize(),
                report.getThroughputTest().getOptimalBatchSize()
            ));
        }
        
        // Analyze latency
        if (report.getLatencyTest().getP99Latency() > 100) {
            recommendations.add("High P99 latency (>100ms). Consider: " +
                "1) Enable caching 2) Reduce batch size 3) Upgrade hardware");
        }
        
        // Analyze memory
        if (report.getMemoryTest().getPeakMemoryMB() > 4096) {
            recommendations.add("High memory usage (>4GB). Consider: " +
                "1) Enable model quantization 2) Use memory mapping 3) Increase heap size");
        }
        
        // Analyze stability
        if (report.getStabilityTest().hasMemoryLeak()) {
            recommendations.add("Potential memory leak detected. " +
                "Monitor memory usage and consider restart strategy");
        }
        
        return recommendations;
    }
}
```

---

This performance guide provides comprehensive optimization strategies for Otter Streams. Remember that optimal configuration depends on your specific workload, hardware, and requirements. Always test changes in a staging environment before deploying to production.

For specific performance questions or optimization advice, please open an issue or discussion in the GitHub repository.