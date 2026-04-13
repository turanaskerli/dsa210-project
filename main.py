import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import berserk
from scipy.stats import chi2_contingency
import os
from dotenv import load_dotenv

# Data Loading
df = pd.read_csv('data/games.csv')
df.dropna(subset=['white_rating', 'black_rating', 'opening_name', 'winner'], inplace=True)

bins = [0, 1200, 1800, 3000]
labels = ['Beginner', 'Intermediate', 'Advanced']
df['rating_bracket'] = pd.cut(df['white_rating'], bins=bins, labels=labels)

# Data Enrichment
load_dotenv() 
API_TOKEN = os.getenv('LICHESS_TOKEN')
session = berserk.TokenSession(API_TOKEN)
client = berserk.Client(session=session)

try:
    enriched_games = client.games.export_by_player('thibault', max=10)
    print("Enrichment connection successful. Data retrieved from Lichess API.")
except Exception as e:
    print(f"API Connection Issue: {e}")

# Hypothesis Testing
print("\n--- Starting Hypothesis Testing ---")

top_10_openings = df['opening_name'].value_counts().nlargest(10).index
df_top_10 = df[df['opening_name'].isin(top_10_openings)]

# Chi-Square Test 
contingency_table = pd.crosstab(df_top_10['opening_name'], df_top_10['rating_bracket'])

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Contingency Table Created.")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

if p < 0.05:
    print("Result: Reject the Null Hypothesis (Statistically Significant).")
else:
    print("Result: Fail to Reject the Null Hypothesis.")

print("--- Hypothesis Testing Complete ---\n")

# EDA
plt.figure(figsize=(12, 6))
sns.countplot(data=df_top_10, x='opening_name', hue='winner')
plt.xticks(rotation=45, ha='right')
plt.title('Win/Loss Distribution for Top 10 Openings')
plt.tight_layout()

# Display the Graph
print("Displaying Graph... (Close the window to end the script)")
plt.show()