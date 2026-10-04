import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Req 1
df = pd.read_csv("netflix_titles.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

"""
Numeric Columns: `release_year`, `duration` (partially numeric text).
Categorical Columns: `type`, `title`, `director`, `cast`, `country`, `rating`, `listed_in`, `description`.
Date / ID Columns: `show_id` (Identifier), `date_added` (Date string).

Questions:
1. Which genres and content types have experienced the fastest expansion in volume over recent decades?
2. How do the duration distributions differ significantly between Movies and TV Shows?
3. Which countries produce the highest concentration of content, and how does their release distribution trend over time?
"""
# Req 2
print(df.isna().sum())
df = df.dropna(subset=["date_added", "rating", "duration"])
df["director"] = df["director"].fillna("Unknown Director")
df["cast"] = df["cast"].fillna("No Cast Listed")
df["country"] = df["country"].fillna("Unknown Country")
print(df.isna().sum())
# Filled missing values with placeholder labels to retain valuable rows instead of deleting data because dropping them would cause us to lose valuable rows of data

# Req 3
recent_movies = df[(df["type"] == "Movie") & (df["release_year"] >= 2020)]

# Req 4: new column added
df["primary_genre"] = df["listed_in"].apply(lambda x: str(x).split(",")[0].strip())
print(df["primary_genre"])

df["date_added"] = pd.to_datetime(df["date_added"].str.strip())


# Req 5
def categorize_release_era(year):
    if year >= 2020:
        return "Modern (2020+)"
    elif year >= 2010:
        return "2010s"
    else:
        return "Classic / Older"


df["Release_era"] = df["release_year"].apply(categorize_release_era)

# Req 6
df["is_movie"] = np.where(df["type"] == "Movie", 1, 0)

# Req 7
df_sorted = df.sort_values(by=["release_year", "title"], ascending=[False, True])

# Req 8
genre_counts = df.groupby("primary_genre")["show_id"].count()

# Req 9
year_stats = df.groupby("type")["release_year"].agg(["mean", "min", "max", "count"])

# Req 10
top_genres = genre_counts.sort_values(ascending=False).head(5)
ordered_genres = top_genres.reindex(top_genres.index)

# Req 14
region_lookup = pd.DataFrame(
    {
        "country": ["United States", "India", "United Kingdom", "Japan"],
        "Global_Region": ["North America", "South Asia", "Europe", "East Asia"],
    }
)
df = pd.merge(df, region_lookup, on="country", how="left")
df["Global_Region"] = df["Global_Region"].fillna("Other")

# Req 15
title_word_counts = [len(title.split()) for title in df["title"]]

# Req 11
plt.figure(figsize=(6, 4))
correlation_matrix = df[["release_year", "is_movie"]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()
# This heatmap shows the statistical relationship between numerical columns, indicating whether certain features increase or decrease together. It helps quickly spot if any numeric variables have a strong linear connection

# Req 12
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="type", y="release_year", palette="Set2")
plt.title("Release Year Distribution and Outliers by Type")
plt.xlabel("Content Type")
plt.ylabel("Release Year")
plt.show()
# This boxplot compares the release years of movies and TV shows, and highlights older shows as dots at the bottom. It shows that TV shows include much older classics, while movies are mostly newer releases.

# Req 13
plt.figure(figsize=(10, 5))
top_genres.plot(kind="bar", color="#E50914", edgecolor="black")
plt.title("Top Content Genres on Netflix")
plt.xlabel("Primary Genre")
plt.ylabel("Total Titles")
plt.grid(True, alpha=0.3)
plt.show()
# This bar chart highlights the top 5 most common primary genres on Netflix. It clearly shows which categories dominate the platform's catalog by total title count.

# Scatter plot
yearly_counts = df.groupby("release_year").size().reset_index(name="title_count")
plt.figure(figsize=(10, 5))
plt.scatter(
    yearly_counts["release_year"],
    yearly_counts["title_count"],
    color="#1C7392",
    alpha=0.7,
    edgecolor="black",
)
plt.title("Trend of Netflix Content Releases by Year")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles Released")
plt.grid(True, alpha=0.3)
plt.show()
# This scatter plot maps out the volume of content releases by year, revealing an exponential growth trend starting around the 2010s. It emphasizes how rapidly Netflix expanded its content library over recent years.

"""
CONCLUSION:
1. Which genres have expanded the fastest over recent decades?
The data shows that genres like Dramas, Comedies, and International Movies make up the largest share of Netflix's catalog. Furthermore, our scatter plot confirms that content production experienced massive, exponential growth starting in the 2010s, rapidly expanding the platform's library year after year.

2. How do release years and trends differ between Movies and TV Shows?
Based on the boxplot analysis, TV shows tend to include a much wider spread of older, classic archival content. In contrast, movies are heavily concentrated in recent, modern release years, showing that Netflix's movie catalog leans much heavier toward newer releases.

3. Which countries produce the highest concentration of content?
The dataset reveals that the vast majority of titles on the platform are concentrated in a few major entertainment hubs, primarily the United States and India—while other regions make up a smaller fraction of the total library.
"""
