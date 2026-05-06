# sembada-cloud


### What is CRISP-ML(Q)
CRISP-ML(Q) stands for Cross-Industry Standard Process for Machine Learning with Quality Assurance methodology. It is a structured, iterative framework designed to guide the entire ML lifecycle, from business understanding to deployment and maintenance, while emphasizing risk management, model quality, and reliability.

Core Phases of CRISP-ML(Q): 
1. Business and Data Understanding: Defines project objectives, translates business goals into ML goals, and assesses project feasibility.
2. Data Engineering: Involves data collection, cleaning, normalization, and feature engineering, often utilizing data lakes, with a focus on data quality.
3. ML Model Engineering: Focuses on algorithm selection, training, hyperparameter tuning, and, when applicable, transfer learning or ensemble methods.
4. ML Model Evaluation: Validates model performance, ensures robustness, checks for fairness and interpretability, and evaluates against success criteria before deployment.
5. Deployment: Releases the model into production systems, including planning for model updates and user adoption.
6. Monitoring and Maintenance: Oversees the model's performance in real-world use, monitoring for data drift and maintaining the system to ensure long-term functionality.

Key Quality Assurance Aspects:
- Risk-Based Thinking: Identifying potential failure points early in the lifecycle.
- Measurable Metrics: Defining clear technical and business metrics for success.
- Iterative Process: Each phase can be revisited, and the model refined based on findings in later stages. 

### What is CAMS DevOps
CAMS is a foundational DevOps framework to define the core pillars of successful DevOps adoption: Culture, Automation, Measurement, and Sharing. It emphasizes balancing human-centric, collaborative processes with technical automation, often aimed at fostering a blameless, transparent, and high-velocity development environment.

Core Pillars of CAMS:
- Culture: Focuses on improving communication, collaboration, and removing silos between software developers and IT operations, often including blameless retrospection.
- Automation: Aims to automate the entire software delivery pipeline—from testing to deployment—reducing manual intervention and speeding up development.
- Measurement: Emphasizes collecting data on metrics related to people, processes, and technology to guide improvements.
- Sharing: Fosters an open environment where knowledge, ideas, and problems are shared to facilitate team learning and collaborative problem-solving

### How to Run

1. Install dependencies:

`pip install -r requirements.txt`

2. Authenticate with GCP:

`gcloud auth application-default login`

3. Set your project ID:

`set GCP_PROJECT_ID=your-gcp-project-id`

4. Run the script (quick test):

`python app/src/fetch_cluster_data.py`

5. Or open the notebook:

`jupyter notebook notebooks/Google_ClusterData2019Traces.ipynb`