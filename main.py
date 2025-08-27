import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_movies = 100
data = {
    'MovieTitle': [f'Movie {i}' for i in range(1, num_movies + 1)],
    'Budget': np.random.randint(50000000, 200000000, num_movies),
    'AvgReviewScore': np.random.uniform(5, 10, num_movies),
    'ReviewerCredibility': np.random.uniform(0.5, 1, num_movies),
    'WeightedAvgScore': np.random.uniform(4, 9, num_movies),
    'BoxOfficeRevenue': np.random.randint(10000000, 1000000000, num_movies)
}
df = pd.DataFrame(data)
#Adding some noise to the weighted score to make it less perfect predictor.
df['WeightedAvgScore'] = df['WeightedAvgScore'] + np.random.normal(0, 0.5, num_movies)
df['WeightedAvgScore'] = df['WeightedAvgScore'].clip(4,9) #Keep it within reasonable range
# --- 2. Data Cleaning and Feature Engineering ---
# No significant cleaning needed for synthetic data, but in real-world scenarios, this would be crucial.
df['WeightedSentiment'] = df['AvgReviewScore'] * df['ReviewerCredibility']
# --- 3. Analysis ---
# Perform linear regression to assess the relationship between weighted sentiment and box office revenue
slope, intercept, r_value, p_value, std_err = linregress(df['WeightedSentiment'], df['BoxOfficeRevenue'])
print(f"Linear Regression Results:")
print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")
print(f"R-squared: {r_value**2:.2f}")
print(f"P-value: {p_value:.3f}")
# --- 4. Visualization ---
plt.figure(figsize=(10, 6))
sns.regplot(x='WeightedSentiment', y='BoxOfficeRevenue', data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Box Office Revenue vs. Weighted Sentiment Score')
plt.xlabel('Weighted Sentiment Score')
plt.ylabel('Box Office Revenue')
plt.grid(True)
plt.tight_layout()
# Save the plot to a file
output_filename = 'box_office_vs_sentiment.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
plt.figure(figsize=(10,6))
sns.histplot(df['WeightedAvgScore'], kde=True)
plt.title('Distribution of Weighted Average Scores')
plt.xlabel('Weighted Average Score')
plt.ylabel('Frequency')
plt.savefig('weighted_score_distribution.png')
print(f"Plot saved to weighted_score_distribution.png")