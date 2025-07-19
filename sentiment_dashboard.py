# sentiment_dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm

# NLTK downloads (run only once)
nltk.download('vader_lexicon')

# Title and Info
st.title("📊 Sentiment Dashboard: VADER & RoBERTa")
st.markdown("Visualizing product review sentiments using two methods: **VADER** (rule-based) and **RoBERTa** (transformer-based).")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Review.csv")
    return df.head(500)  # Limit to 500 rows

df = load_data()
st.success("✅ Reviews Loaded!")
st.write("**Dataset Shape:**", df.shape)
st.dataframe(df[['Id', 'Text', 'Score']].sample(5))

# Plot: Count of Reviews by Score
st.subheader("📌 Review Count by Score")
fig1, ax1 = plt.subplots()
df['Score'].value_counts().sort_index().plot(kind='bar', color='skyblue', ax=ax1)
ax1.set_xlabel("Review Score")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# VADER sentiment analysis
st.subheader("🧠 VADER Sentiment Analysis")
sia = SentimentIntensityAnalyzer()

@st.cache_data
def run_vader(df):
    vader_scores = {}
    for i, row in df.iterrows():
        score = sia.polarity_scores(row['Text'])
        vader_scores[row['Id']] = score
    vaders = pd.DataFrame(vader_scores).T.reset_index().rename(columns={'index': 'Id'})
    return vaders.merge(df, on='Id')

vader_df = run_vader(df)

# Plot: Compound sentiment vs. Score
fig2, ax2 = plt.subplots()
sns.barplot(data=vader_df, x='Score', y='compound', color='orange', ax=ax2)
ax2.set_title("Compound Sentiment by Score")
st.pyplot(fig2)

# Positive / Neutral / Negative sentiment
fig3, axs = plt.subplots(1, 3, figsize=(15, 4))
sns.barplot(data=vader_df, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vader_df, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vader_df, x='Score', y='neg', ax=axs[2])
axs[0].set_title("Positive")
axs[1].set_title("Neutral")
axs[2].set_title("Negative")
st.pyplot(fig3)

# RoBERTa sentiment scoring
st.subheader("🤖 RoBERTa Sentiment Analysis")
@st.cache_resource
def load_roberta_model():
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return model, tokenizer

model, tokenizer = load_roberta_model()

@st.cache_data
def run_roberta(df):
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        encoded = tokenizer(row['Text'], return_tensors='pt', truncation=True)
        with torch.no_grad():
            output = model(**encoded)
        scores = softmax(output[0][0].numpy())
        results.append({
            'Id': row['Id'],
            'roberta_neg': scores[0],
            'roberta_neu': scores[1],
            'roberta_pos': scores[2]
        })
    return pd.DataFrame(results)

roberta_df = run_roberta(df)
merged = pd.merge(vader_df, roberta_df, on='Id')

# Pairplot
st.subheader("📈 Sentiment Comparison (VADER vs RoBERTa)")
st.markdown("Comparing sentiment scores between models")
fig4 = sns.pairplot(data=merged,
                    vars=['vader_neg', 'vader_neu', 'vader_pos', 'roberta_neg', 'roberta_neu', 'roberta_pos'],
                    hue='Score',
                    palette='tab10')
st.pyplot(fig4)
