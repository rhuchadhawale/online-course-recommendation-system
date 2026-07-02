# online-course-recommendation-system
A hybrid course recommendation engine combining content-based filtering (TF-IDF + Nearest Neighbors) and item-based collaborative filtering (cosine similarity), deployed as an interactive Streamlit app. Recommends relevant online courses based on course content, ratings, and user behavior.
<img width="940" height="591" alt="Screenshot 2026-07-02 121856" src="https://github.com/user-attachments/assets/81edaab5-eddf-
📌 Overview

This project builds and compares three recommendation strategies on a dataset of 100,000 user-course interactions across 20 courses and 43,000+ users:


Content-Based Filtering — Recommends courses similar in content (name via TF-IDF) and attributes (price, rating, difficulty, enrollment) using Nearest Neighbors with cosine distance.
Item-Based Collaborative Filtering — Recommends courses based on similarity in user rating patterns, using cosine similarity on a course–user rating matrix.
Hybrid Model — Merges recommendations from both approaches, removing duplicates, to improve overall relevance.


All three models are evaluated and compared using Precision@5, Recall@5, and F1 Score.


🛠️ Tech Stack


Language: Python
Data Processing: Pandas, NumPy
Machine Learning: Scikit-learn (TF-IDF, StandardScaler, NearestNeighbors, cosine similarity)
Visualization: Matplotlib, Seaborn
Deployment: Streamlit
Sparse Matrix Ops: SciPy



📊 Dataset


100,000 rows of user-course interaction data
20 unique courses, 43,242 unique users
Features include: course name, instructor, duration, difficulty level, rating, enrollment numbers, price, feedback score, study material availability, and time spent


Data was cleaned and feature-engineered (binary encoding for certification/study material, ordinal encoding for difficulty, average course rating aggregation) before modeling.


🧠 How It Works


Content-Based Filtering
Course names are vectorized with TF-IDF; numeric features (price, rating, difficulty, enrollment) are scaled and combined into a single feature matrix. A Nearest Neighbors model (cosine metric) finds the closest courses to a selected course.
Item-Based Collaborative Filtering
A course–user rating matrix is built, and cosine similarity between courses is computed to recommend courses that tend to be rated similarly by the same users.
Hybrid Recommendation
Combines the top results from both models into a single deduplicated list, balancing content similarity with collaborative signal.
📈 Model Evaluation

Each model was evaluated on a sample of courses using Precision@5, Recall@5, and F1 Score, with relevance defined by matching difficulty level.

├── app.py                                  # Streamlit application
├── onlinecourserecommendation.ipynb        # EDA, feature engineering, model building & evaluation
├── final_course_dataset.csv                # Processed dataset
├── requirements.txt                        # Python dependencies
└── README.md

✨ Features
Interactive course selection with three switchable recommendation models
Real-time recommendations via Streamlit UI
Dataset statistics dashboard (total records, courses, users)
Visual comparison of recommendation model performance

 Future Improvements
Incorporate matrix factorization (SVD) for deeper collaborative filtering
Add user-based collaborative filtering alongside item-based
Deploy with a larger, more diverse course catalog
A/B test hybrid weighting strategies

👤 Author
Rhucha Dhawale
