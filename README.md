# Best EU LEAGUE
# Football League Ranking Prediction

This project aims to predict the ranking of football leagues (such as Premier League, Bundesliga, Serie A, and Ligue 1) using various team statistics. The following steps outline the data preprocessing, model training, hyperparameter tuning, and evaluation process.
We only have 4 leagues. Data from other seasons is not available on the official LaLiga website.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Project Overview](#project-overview)
- [Steps in Detail](#steps-in-detail)
  - [1. Importing Libraries](#1-importing-libraries)
  - [2. Defining Constants](#2-defining-constants)
  - [3. Defining Functions](#3-defining-functions)
  - [4. Scraping Data](#4-scraping-data)
    - [4.1 Ligue 1](#41-ligue-1)
    - [4.2 Bundesliga](#42-bundesliga)
    - [4.3 Serie A](#43-serie-a)
    - [4.4 Premier League](#44-premier-league)
  - [5. Preprocessing](#5-preprocessing)
  - [6. Data Visualization](#6-data-visualization)
  - [7. Feature Engineering](#7-feature-engineering)
  - [8. Hyperparameter Tuning and Model Evaluation](#8-hyperparameter-tuning-and-model-evaluation)
- [Conclusion](#conclusion)

## Prerequisites

- Python 3.x
- Libraries: `numpy`, `pandas`, `scikit-learn`, `BeautifulSoup`, `requests`, `matplotlib`, `seaborn`

Install the required libraries using pip:
```bash
pip install numpy pandas scikit-learn beautifulsoup4 requests matplotlib seaborn
```

## Project Overview

The goal of this project is to predict the ranking of a football league based on team statistics. The workflow includes data scraping, preprocessing, model training, hyperparameter optimization, and evaluation.

## Steps in Detail

### 1. Importing Libraries

```python
from bs4 import BeautifulSoup as BS
import os, os.path as path
import requests as request
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
```

**Why?**
- These libraries provide tools for web scraping (`BeautifulSoup`, `requests`), data manipulation (`numpy`, `pandas`), preprocessing (`LabelEncoder`, `OneHotEncoder`, `StandardScaler`), visualization (`matplotlib`, `seaborn`), modeling (`RandomForestRegressor`), and performance evaluation (`r2_score`, `mean_squared_error`).

### 2. Defining Constants

```python
COLUMNS = ['id', 'team', 'point', 'played', 'won', 'drawn', 'lost', 'goal_scored', 'goal_conceded', 'goal_diff']
SEASONS = {
    '2022-2023':'2022-23',
    '2021-2022':'2021-22',
    '2020-2021':'2020-21',
    '2019-2020':'2019-20',
    '2018-2019':'2018-19'
}
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) \
                    AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
}
```

**Why?**
- These constants define the columns for the data, the seasons of interest, and the headers for the web scraping requests to mimic a real browser.

### 3. Defining Functions

```python
def append_team_data(team_list, id, name, point, played, won, drawn, lost, goal_scored, goal_conceded, goal_diff):
    team = []
    team.append(id)
    team.append(name)
    team.append(point)
    team.append(played)
    team.append(won)
    team.append(drawn)
    team.append(lost)
    team.append(goal_scored)
    team.append(goal_conceded)
    team.append(goal_diff)

    team_list.append(team)
```

**Why?**
- The `append_team_data` function helps in appending the scraped team data into a list.

### 4. Scraping Data

#### 4.1 Ligue 1

```python
params = {
    'StatsActiveTab': 0
}

for season in SEASONS.keys():
  team_list = []
  params['seasonId'] = season
  response = request.get(ligue_1_url, params= params, headers= HEADERS)
  if response.ok:
    soup= BS(response.content, 'lxml')
    li_list = soup.find('div', class_='classement-table-body').find_all('li')

    for li in li_list:
      div_list = li.find_all('div')
      id = int(div_list[0].string)
      name = str.upper(div_list[1].span.string)
      point = int(div_list[2].string)
      played = int(div_list[3].string)
      won = int(div_list[4].string)
      drawn = int(div_list[5].string)
      lost = int(div_list[6].string)
      goal_scored = int(div_list[7].string)
      goal_conceded = int(div_list[8].string)
      goal_diff = int(div_list[9].string)

      append_team_data(team_list, id, name, point, played, won, drawn, lost, goal_scored, goal_conceded, goal_diff)

  df = pd.DataFrame(team_list, columns=COLUMNS)
  df.to_csv(path.join(dataset_brut_url+'/ligue1', season+'.csv'), header=True, index= False)
```

#### 4.2 Bundesliga

```python
for season in SEASONS.keys():
  team_list = []
  response = request.get(bundesliga_url+f"/{season}", headers= HEADERS)
  if response.ok:
    soup= BS(response.content, 'lxml')
    tr_list = soup.tbody.find_all('tr')
    for tr in tr_list:
      td_list = tr.find_all('td')
      id = int(td_list[1].span.string)
      name = str.upper(td_list[3].div['title'])
      point = int(td_list[11].span.string)
      played = int(td_list[5].span.string)
      won = int(td_list[6].span.string)
      drawn = int(td_list[7].span.string)
      lost = int(td_list[8].span.string)
      gs, gc = td_list[9].string.split(':')
      goal_scored = int(gs)
      goal_conceded = int(gc)
      goal_diff = int(td_list[10].span.string)

      append_team_data(team_list, id, name, point, played, won, drawn, lost, goal_scored, goal_conceded, goal_diff)

  df = pd.DataFrame(team_list, columns=COLUMNS)
  df.to_csv(path.join(dataset_brut_url+'/bundesliga', season+'.csv'), header=True, index= False)
```

#### 4.3 Serie A

```python
params = {
    'CAMPIONATO': 'A',
    'TURNO': 'UNICO',
    'GIRONE': 'UNI'
}

for season, param in SEASONS.items():
  team_list = []
  params['STAGIONE'] = param
  response = request.get(serie_a_api, params= params, headers= HEADERS)
  if response.ok:
    data = json.loads(response.text)['data']
    for team in data:
      id = team['PosCls']
      name = team['Nome']
      point = team['PuntiCls']
      played = team['Giocate']
      won = team['Vinte']
      drawn = team['Pareggiate']
      lost = team['Perse']
      goal_scored = team['RETIFATTE']
      goal_conceded = team['RETISUBITE']
      goal_diff = team['RETIFATTE'] - team['RETISUBITE']

      append_team_data(team_list, id, name, point, played, won, drawn, lost, goal_scored, goal_conceded, goal_diff)

  df = pd.DataFrame(team_list, columns=COLUMNS)
  df.to_csv(path.join(dataset_brut_url+'/seriea', season+'.csv'), header=True, index= False)
```

#### 4.4 Premier League

```python
params = {
    'altIds': True,
    'detail': 2,
    'FOOTBALL_COMPETITION': 1
}
seasons = {
    '2022-2023':489,
    '2021-2022':418,
    '2020-2021':363,
    '2019-2020':274,
    '2018-2019':210
}
header = {
    'origin': 'https://www.premierleague.com'
}

for season, param in seasons.items():
  team_list = []
  params['compSeasons'] = param
  response = request.get(premier_league_api

, params= params, headers= header)
  if response.ok:
    data = json.loads(response.text)['tables'][0]['entries']
    for team in data:
      id = team['position']
      name = str.upper(team['team']['name'])
      team = team['overall']
      point = team['points']
      played = team['played']
      won = team['won']
      drawn = team['drawn']
      lost = team['lost']
      goal_scored = team['goalsFor']
      goal_conceded = team['goalsAgainst']
      goal_diff = team['goalsDifference']

      append_team_data(team_list, id, name, point, played, won, drawn, lost, goal_scored, goal_conceded, goal_diff)

  df = pd.DataFrame(team_list, columns=COLUMNS)
  df.to_csv(path.join(dataset_brut_url+'/premierleague', season+'.csv'), header=True, index= False)
```

### 5. Preprocessing

```python
def load_preprocess_league(league_folder):
    league_dfs = []
    for season_file in os.listdir(league_folder):
        if season_file.endswith('.csv'):
            season_path = os.path.join(league_folder, season_file)
            df = pd.read_csv(season_path)

            # Convert numeric columns to float
            numeric_cols = ['point', 'played', 'won', 'drawn', 'lost', 'goal_scored', 'goal_conceded', 'goal_diff']
            df[numeric_cols] = df[numeric_cols].astype(float)

            # Data cleaning
            df = df.dropna(subset=['team'])  # Drop rows with missing team names
            df = df.drop_duplicates()  # Remove duplicate rows

            # Data transformation
            df['win_percentage'] = (df['won'] / df['played']).round(6)
            df['loss_percentage'] = (df['lost'] / df['played']).round(6)
            df['goals_per_game'] = (df['goal_scored'] / df['played']).round(6)
            df['goals_against_per_game'] = (df['goal_conceded'] / df['played']).round(6)
            df = df.drop(columns=['id'])

            league_dfs.append(df)

    league_df = pd.concat(league_dfs, ignore_index=True)
    return league_df

# Load and preprocess data for all leagues
all_leagues = []
for league_folder in ['bundesliga', 'ligue1', 'premierleague', 'seriea']:
    league_path = os.path.join(dataset_brut_url, league_folder)
    league_df = load_preprocess_league(league_path)
    league_df['league'] = league_folder
    all_leagues.append(league_df)

# Concatenate data from all leagues
combined_df = pd.concat(all_leagues, ignore_index=True)

print(combined_df.isnull().sum())

combined_df
```

**Why?**
- The `load_preprocess_league` function loads, cleans, and preprocesses the data for each league, converting numeric columns to float, dropping missing and duplicate rows, and adding new features like win percentage and goals per game.

### 6. Data Visualization

```python
# Calculate average goals scored per game for each league
avg_goals_per_game = combined_df.groupby('league')['goals_per_game'].mean().reset_index()

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x='league', y='goals_per_game', data=avg_goals_per_game, palette='viridis')
plt.xlabel('League')
plt.ylabel('Average Goals Scored Per Game')
plt.title('Comparison of League Performances (Goals Scored Per Game)')
plt.xticks(rotation=45)
plt.show()

# Calculate average win percentage for each league
avg_win_percentage = combined_df.groupby('league')['win_percentage'].mean().reset_index()

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x='league', y='win_percentage', data=avg_win_percentage, palette='magma')
plt.xlabel('League')
plt.ylabel('Average Win Percentage')
plt.title('Comparison of League Performances (Win Percentage)')
plt.xticks(rotation=45)
plt.show()

# Define figure size
plt.figure(figsize=(12, 8))

# Create a boxplot for each league
sns.boxplot(x='league', y='goal_diff', data=combined_df, palette='Set3')
plt.xlabel('League')
plt.ylabel('Goal Difference')
plt.title("Gap between Teams in Each League (Goal Difference)")
plt.xticks(rotation=45)
plt.show()
```

**Why?**
- These visualizations help compare the average goals scored per game, win percentage, and goal difference among different leagues.

### 7. Feature Engineering

```python
label_encoder = LabelEncoder()
combined_df['team'] = label_encoder.fit_transform(combined_df['team'])

categorical_cols = ['league', 'team']
numeric_cols = ['point', 'played', 'won', 'drawn', 'lost', 'goal_scored', 'goal_conceded', 'goal_diff',
                'win_percentage', 'loss_percentage', 'goals_per_game', 'goals_against_per_game']

one_hot_encoder = OneHotEncoder(sparse=False)
combined_df_encoded = one_hot_encoder.fit_transform(combined_df[categorical_cols])

scaler = StandardScaler()
combined_df_numeric = scaler.fit_transform(combined_df[numeric_cols])

X = np.concatenate([combined_df_encoded, combined_df_numeric], axis=1)

# Define the target variable
league_rankings = {'premierleague': 1, 'bundesliga': 2, 'seriea': 3, 'ligue1': 4}
combined_df['league_rank'] = combined_df['league'].map(league_rankings)

# Split the data into features and target
X = combined_df.drop(['league', 'league_rank'], axis=1)
y = combined_df['league_rank']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Why?**
- This step encodes categorical variables, scales numerical variables, and concatenates them to create the feature matrix. It also defines the target variable and splits the data into training and testing sets.

### 8. Hyperparameter Tuning and Model Evaluation

```python
# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print("Best hyperparameters:", grid_search.best_params_)
print("Best cross-validation score:", -grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

predicted_league_rankings = {1: 'premierleague', 2: 'bundesliga', 3: 'seriea', 4: 'ligue1'}
predicted_league = predicted_league_rankings[int(round(y_pred.mean()))]
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
print("The best league is:", predicted_league)

print("Y pred mean:", y_pred.mean())
```

**Why?**
- This step involves hyperparameter tuning using GridSearchCV, evaluating the best model, and predicting the league ranking. The model's performance is measured using Mean Squared Error (MSE) and R-squared (R2) metrics.

## Conclusion

This project successfully demonstrates how to predict the ranking of football leagues using team statistics. The process involves data scraping, preprocessing, feature engineering, model training, and evaluation. The Random Forest Regressor model was used, and hyperparameter tuning was performed to optimize the model's performance.

```
