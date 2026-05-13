Here's a comprehensive list of all algorithms and models mentioned in Chapter 2, organized by category:

**Classical / Statistical Models**
- ARIMA (AutoRegressive Integrated Moving Average)

**Machine Learning Models**
- Random Forest
- XGBoost
- CatBoost (gradient boosting, used for AWS EC2 price prediction)
- K-Means Clustering (for workload segmentation)
- Regression models (general, for demand forecasting)

**Deep Learning Architectures**
- MLP (Multi-Layer Perceptron)
- RNN (Recurrent Neural Network)
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- BiGRU (Bidirectional GRU)
- CNN (Convolutional Neural Network)
- CNN-LSTM (hybrid architecture)
- DWT-BiGRU (Discrete Wavelet Transformation + BiGRU + Attention Mechanism)
- Temporal Fusion Transformer (TFT)
- Autoencoders (for anomaly/cost spike detection)

**Reinforcement Learning**
- Q-Learning (used in RLPRAF framework)
- Deep Reinforcement Learning (DRL)
- Federated Reinforcement Learning (privacy-preserving multi-cloud)

**Evolutionary / Hybrid Optimization**
- GA-PSO (Genetic Algorithm + Particle Swarm Optimization)
- FLGAPSONN (Functional Link Neural Network trained via GA-PSO)
- SDWF — Self-Directed Workload Forecasting (uses Blackhole heuristic for self-correction)

**Anomaly Detection Methods**
- Weighted Hybrid Algorithm (WHA) — combines Moving Median, Kalman Filter, and Savitzky-Golay Filter
- Isolation Forest
- Clustering-based detection

**Explainability / Interpretability (XAI)**
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Integrated Gradients

**Named Frameworks/Systems (model pipelines)**
- CRUOS — Cloud Resource Usage Optimization System
- MSFS — Multi-Time Series Forecasting System (with STG grouping method)
- RLPRAF — Reinforcement Learning-Based Proactive Resource Allocation Framework
- ARANDO — Adaptive Resource Allocation and Optimization framework

The chapter's strongest recommendations lean toward **LSTM, BiGRU, TFT, and CNN-LSTM** for forecasting, **WHA** for anomaly preprocessing, **SHAP/Integrated Gradients** for explainability, and **Q-Learning/DRL** for proactive auto-scaling.