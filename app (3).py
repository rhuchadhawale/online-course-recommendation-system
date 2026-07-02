import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Online Course Recommendation System",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🎓 Online Course Recommendation System")

st.markdown("""
### About the Project

This system recommends online courses using:

- 📚 Content-Based Filtering
- 🤝 Item-Based Collaborative Filtering
- 🚀 Hybrid Recommendation

Select a course and recommendation model to discover relevant courses.
""")

st.markdown("---")

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("final_course_dataset.csv")

# -----------------------------
# Content Based Preparation
# -----------------------------

course_df = df.drop_duplicates(
    subset='course_name'
).reset_index(drop=True)

course_df['course_name'] = course_df['course_name'].fillna('Unknown')

numeric_features = course_df[
    [
        'course_price',
        'avg_course_rating',
        'study_material_binary',
        'difficulty_encoded',
        'enrollment_numbers'
    ]
]

tfidf = TfidfVectorizer(stop_words='english')

course_text = tfidf.fit_transform(
    course_df['course_name']
)

scaler = StandardScaler()

numeric_scaled = scaler.fit_transform(
    numeric_features
)

content_matrix = hstack(
    [
        course_text,
        numeric_scaled
    ]
).tocsr()

content_model = NearestNeighbors(
    metric='cosine',
    algorithm='brute'
)

content_model.fit(content_matrix)

# -----------------------------
# Collaborative Filtering
# -----------------------------

course_user = df.pivot_table(
    index='course_name',
    columns='user_id',
    values='rating',
    fill_value=0
)

course_similarity = cosine_similarity(
    course_user
)

course_similarity_df = pd.DataFrame(
    course_similarity,
    index=course_user.index,
    columns=course_user.index
)

# -----------------------------
# Content Recommendation
# -----------------------------

def recommend_courses(course_name, top_n=5):

    matches = course_df[
        course_df['course_name'] == course_name
    ].index

    if len(matches) == 0:
        return pd.DataFrame()

    idx = matches[0]

    distances, indices = content_model.kneighbors(
        content_matrix[idx],
        n_neighbors=min(top_n + 1, len(course_df))
    )

    recommendations = course_df.iloc[
        indices[0][1:]
    ]

    return recommendations[
        [
            'course_name',
            'avg_course_rating',
            'course_price',
            'difficulty_level'
        ]
    ]

# -----------------------------
# Item Based Recommendation
# -----------------------------

def similar_courses(course_name, top_n=5):

    if course_name not in course_similarity_df.index:
        return pd.DataFrame()

    similar = (
        course_similarity_df.loc[course_name]
        .sort_values(ascending=False)
        .iloc[1:top_n+1]
    )

    return pd.DataFrame({
        'Course': similar.index,
        'Similarity Score': similar.values
    })

# -----------------------------
# Hybrid Recommendation
# -----------------------------

def hybrid_recommend(course_name, top_n=5):

    content_recs = recommend_courses(
        course_name,
        top_n
    )

    item_recs = similar_courses(
        course_name,
        top_n
    )

    content_list = list(
        content_recs['course_name']
    )

    item_list = list(
        item_recs['Course']
    )

    combined = []

    for course in content_list + item_list:

        if course not in combined:
            combined.append(course)

    return pd.DataFrame({
        'Recommended Course':
        combined[:top_n]
    })

# -----------------------------
# Streamlit UI
# -----------------------------

# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Recommendation Settings")

model = st.sidebar.selectbox(
    "Select Recommendation Model",
    [
        "Content-Based",
        "Item-Based",
        "Hybrid"
    ]
)

# --------------------------------------------------
# COURSE SELECTION
# --------------------------------------------------

course_name = st.selectbox(
    "📖 Select Course",
    sorted(
        course_df["course_name"].unique()
    )
)

if st.button("Recommend"):

    if model == "Content-Based":

        st.subheader(
            "Content-Based Recommendations"
        )

        st.dataframe(
            recommend_courses(course_name)
        )

    elif model == "Item-Based":

        st.subheader(
            "Item-Based Recommendations"
        )

        st.dataframe(
            similar_courses(course_name)
        )

    else:

        st.subheader(
            "Hybrid Recommendations"
        )

        st.dataframe(
            hybrid_recommend(course_name)
        )

st.markdown("---")

st.subheader("Dataset Statistics")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Records",
    len(df)
)

col2.metric(
    "Courses",
    df['course_name'].nunique()
)

col3.metric(
    "Users",
    df['user_id'].nunique()
)

