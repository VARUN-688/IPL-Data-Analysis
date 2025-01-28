#!/usr/bin/env python
# coding: utf-8

# In[122]:


import pandas as pd


# In[123]:


deliveries=pd.read_csv("deliveries_cleaned_data.csv",index_col=0)


# In[124]:


deliveries.head()


# In[125]:


matches=pd.read_csv("matches_cleaned_data.csv",index_col=0)


# In[126]:


matches.head()


# In[127]:


batsman_avg=deliveries.groupby('batter',as_index=False).agg({"batsman_runs":'sum','match_id':'nunique'})
batsman_avg.sort_values(by=['batsman_runs','match_id'],ascending=[False,True],inplace=True)


# In[128]:


batsman_avg['Average']=round(batsman_avg['batsman_runs']/batsman_avg['match_id'],2)


# In[129]:


import matplotlib.pyplot as plt


# In[130]:


fig, ax = plt.subplots(2, 1, figsize=(12, 10))

batsmen = batsman_avg['batter'][:10][::-1]
runs = batsman_avg['batsman_runs'][:10][::-1]
averages = batsman_avg['Average'][:10][::-1]

ax[0].barh(batsmen, runs)
ax[0].set_title("Top 10 Batsmen by Runs")
ax[0].set_xlabel("Runs Scored")
ax[0].set_ylabel("Batsman")
for i, v in enumerate(runs):
    ax[0].text(v + 5, i, str(v), va='center')

ax[1].barh(batsmen, averages)
ax[1].set_title("Average of Top 10 Batsmen by Runs")
ax[1].set_xlabel("Average Runs Scored")
ax[1].set_ylabel("Batsman")
for i, v in enumerate(averages):
    ax[1].text(v + 0.1, i, str(v), va='center')

plt.tight_layout()
plt.show()


# In[131]:


team_avg=deliveries.groupby('batting_team',as_index=False).agg({'batsman_runs':'sum','match_id':'nunique'})


# In[132]:


team_avg.sort_values(by=['batsman_runs','match_id'],ascending=[False,True],inplace=True)


# In[133]:


team_avg.reset_index(inplace=True)


# In[134]:


team_avg['Average']=round(team_avg['batsman_runs']/team_avg['match_id'],2)


# In[135]:


fig, ax = plt.subplots(2, 1, figsize=(12, 10))

team = team_avg['batting_team'][::-1]
runs = team_avg['batsman_runs'][::-1]
averages = team_avg['Average'][::-1]

ax[0].barh(team, runs)
ax[0].set_title("Top Teams by Runs")
ax[0].set_xlabel("Runs Scored")
ax[0].set_ylabel("Team")
for i, v in enumerate(runs):
    ax[0].text(v + 5, i, str(v), va='center')

ax[1].barh(team, averages)
ax[1].set_title("Average of Teams by Runs")
ax[1].set_xlabel("Average Runs Scored")
ax[1].set_ylabel("Team")
for i, v in enumerate(averages):
    ax[1].text(v + 0.1, i, str(v), va='center')

plt.tight_layout()
plt.show()


# In[136]:


batter=deliveries.groupby(['batter','batting_team','match_id'],as_index=False).agg(
    {'batsman_runs':'sum','is_wicket':'sum',
    'ball':'count'
    })


# In[137]:


batter


# In[138]:


batter['strike_rate']=batter['batsman_runs']/batter['ball']


# In[139]:


batter


# In[140]:


batter.sort_values(by='batsman_runs',ascending=False)


# In[141]:


batter=pd.merge(batter,matches.loc[:,['id','season','date','venue']], how='left',left_on='match_id',right_on='id')


# In[142]:


batter


# In[143]:


batter_pivot=batter.pivot_table(index=['batter','batting_team'],columns=['season'],values=['batsman_runs'],aggfunc='sum')


# In[144]:


best_batsman=batter.loc[batter['batter'].isin(batsman_avg['batter'][:10])]


# In[145]:


best_batsman['date'] = pd.to_datetime(best_batsman['date'])
best_batsman.loc[:, 'year'] = best_batsman['date'].dt.year

agg_runs = best_batsman.groupby(['batter', 'season','batting_team'])['batsman_runs'].sum().reset_index()

unique_batters = agg_runs['batter'].unique()
num_batters = len(unique_batters)

fig, axes = plt.subplots(nrows=num_batters, ncols=1, figsize=(10, num_batters * 4), sharex=True)


for ax, batter_ in zip(axes, unique_batters):
    batter_data = agg_runs[agg_runs['batter'] == batter_]
    for team in batter_data['batting_team'].unique():
        team_data = batter_data[batter_data['batting_team'] == team]
        ax.plot(
            team_data['season'],
            team_data['batsman_runs'],
            marker='o',
            label=f"{team}"
        )
    ax.set_title(f"Performance of {batter_} Over Years", fontsize=14)
    ax.set_ylabel("Total Runs", fontsize=12)
    ax.grid(True)
    ax.legend(title="Batting Team", fontsize=12)

plt.xlabel("Year", fontsize=14)
plt.tight_layout()

plt.show()


# In[146]:


batter


# In[147]:


batter['date']=pd.to_datetime(batter.loc[:,'date'])


# In[148]:


batter['year']=batter.loc[:,'date'].dt.year


# In[149]:


top_each_season=batter.groupby(['season','batter'])['batsman_runs'].sum().reset_index().sort_values(by=['season','batsman_runs'],ascending=[True,False])


# In[150]:


top_each_season


# In[151]:


top_each_season['Rank']=top_each_season.groupby('season')['batsman_runs'].rank(method='first',ascending=False)


# In[152]:


top_each_season


# In[153]:


top_each_season.loc[top_each_season['Rank']==1.0]


# In[156]:


best_batsman_each_season=pd.merge(agg_runs,top_each_season,left_on=['batter','season'],right_on=['batter','season'])


# In[157]:


best_batsman_each_season


# In[159]:


unique_batters = best_batsman_each_season['batter'].unique()
num_batters = len(unique_batters)
fig, axes = plt.subplots(nrows=num_batters, ncols=1, figsize=(10, num_batters * 4), sharex=True)

for ax, batter_ in zip(axes, unique_batters):
    batter_data = best_batsman_each_season[best_batsman_each_season['batter'] == batter_]
    for team in batter_data['batting_team'].unique():
        team_data = batter_data[batter_data['batting_team'] == team]
        ax.plot(
            team_data['season'],
            team_data['Rank'],
            marker='o',
            label=f"{team}"
        )
    ax.set_title(f"Performance of {batter_} Over Years", fontsize=14)
    ax.set_ylabel("Total Runs", fontsize=12)
    ax.grid(True)
    ax.legend(title="Batting Team", fontsize=12)

plt.xlabel("Year", fontsize=14)
plt.tight_layout()

plt.show()


# In[160]:


best_batsman


# In[161]:


avg_strike_rate = best_batsman.groupby(['batter', 'year','batting_team'])['strike_rate'].mean().reset_index()
unique_batters = agg_runs['batter'].unique()
num_batters = len(unique_batters)

fig, axes = plt.subplots(nrows=num_batters, ncols=1, figsize=(10, num_batters * 4), sharex=True)


for ax, batter_ in zip(axes, unique_batters):
    batter_data = avg_strike_rate[agg_runs['batter'] == batter_]
    for team in batter_data['batting_team'].unique():
        team_data = batter_data[batter_data['batting_team'] == team]
        ax.plot(
            team_data['year'],
            team_data['strike_rate'],
            marker='o',
            label=f"{team}"
        )
    ax.set_title(f"Average Strike Rate of {batter_} Over Years", fontsize=14)
    ax.set_ylabel("Total Runs", fontsize=12)
    ax.grid(True)
    ax.legend(title="Batting Team", fontsize=12)

plt.xlabel("Year", fontsize=14)
plt.tight_layout()

plt.show()


# In[162]:


boller_avg=deliveries.groupby('bowler',as_index=False).agg({"is_wicket":'sum','match_id':'nunique'})
boller_avg.sort_values(by=['is_wicket','match_id'],ascending=[False,True],inplace=True)


# In[163]:


boller_avg


# In[164]:


boller_avg['Average']=round(boller_avg['is_wicket']/boller_avg['match_id'],2)


# In[165]:


fig, ax = plt.subplots(2, 1, figsize=(12, 10))

boller = boller_avg['bowler'][:10][::-1]
wickets = boller_avg['is_wicket'][:10][::-1]
averages = boller_avg['Average'][:10][::-1]

ax[0].barh(boller, wickets)
ax[0].set_title("Top 10 Bowlers by Wickets")
ax[0].set_xlabel("Wickets Taken")
ax[0].set_ylabel("Bowler")
for i, v in enumerate(wickets):
    ax[0].text(v + 3, i, str(v), va='center')

ax[1].barh(boller, averages)
ax[1].set_title("Average of Top 10 Bowlers by Wickets")
ax[1].set_xlabel("Average Wickets Taken")
ax[1].set_ylabel("Batsman")
for i, v in enumerate(averages):
    ax[1].text(v , i, str(v), va='center')

plt.tight_layout()
plt.show()


# In[166]:


team_bowling_avg=deliveries.groupby('bowling_team',as_index=False).agg({'is_wicket':'sum','match_id':'nunique'})


# In[167]:


team_bowling_avg.sort_values(by=['is_wicket','match_id'],ascending=[False,True],inplace=True)


# In[168]:


team_bowling_avg.reset_index(inplace=True)


# In[169]:


team_bowling_avg['Average']=round(team_bowling_avg['is_wicket']/team_avg['match_id'],2)


# In[170]:


fig, ax = plt.subplots(2, 1, figsize=(12, 10))

team = team_bowling_avg['bowling_team'][::-1]
wickets = team_bowling_avg['is_wicket'][::-1]
averages = team_bowling_avg['Average'][::-1]

ax[0].barh(team, wickets)
ax[0].set_title("Top Teams by Wickets")
ax[0].set_xlabel("Wickets Taken")
ax[0].set_ylabel("Team")
for i, v in enumerate(wickets):
    ax[0].text(v , i, str(v), va='center')

ax[1].barh(team, averages)
ax[1].set_title("Average of Teams by Wickets")
ax[1].set_xlabel("Average Wickets Taken")
ax[1].set_ylabel("Team")
for i, v in enumerate(averages):
    ax[1].text(v + 0.1, i, str(v), va='center')

plt.tight_layout()
plt.show()


# In[171]:


boller=deliveries.groupby(['bowler','bowling_team','match_id'],as_index=False).agg(
    {'is_wicket':'sum','batsman_runs':'sum',
    'ball':'count'
    })


# In[172]:


boller


# In[173]:


boller['economy']=(boller.loc[:,'batsman_runs']/boller.loc[:,'ball'])*6


# In[174]:


boller


# In[175]:


boller=pd.merge(boller,matches.loc[:,['id','season','date','venue']], how='left',left_on='match_id',right_on='id')


# In[176]:


boller_pivot=boller.pivot_table(index=['bowler','bowling_team'],columns=['season'],values=['is_wicket'],aggfunc='sum')
boller_pivot


# In[177]:


best_boller=boller.loc[boller['bowler'].isin(boller_avg['bowler'][:10])]
best_boller


# In[178]:


best_boller['date'] = pd.to_datetime(best_boller['date'])
best_boller.loc[:, 'year'] = best_boller['date'].dt.year

avg_wickets = best_boller.groupby(['bowler', 'season','bowling_team'])['is_wicket'].sum().reset_index()

unique_boller = avg_wickets['bowler'].unique()
num_boller= len(unique_boller)

fig, axes = plt.subplots(nrows=num_boller, ncols=1, figsize=(10, num_boller * 4), sharex=True)


for ax, boller_ in zip(axes, unique_boller):
    boller_data = avg_wickets[avg_wickets['bowler'] == boller_]
    for team in boller_data['bowling_team'].unique():
        team_data = boller_data[boller_data['bowling_team'] == team]
        ax.plot(
            team_data['season'],
            team_data['is_wicket'],
            marker='o',
            label=f"{team}"
        )
    ax.set_title(f"Performance of {boller_} Over Years", fontsize=14)
    ax.set_ylabel("Total Wickets", fontsize=12)
    ax.grid(True)
    ax.legend(title="Bowling Team", fontsize=12)

plt.xlabel("Year", fontsize=14)
plt.tight_layout()

plt.show()


# In[179]:


boller['date']=pd.to_datetime(boller.loc[:,'date'])


# In[180]:


boller['year']=boller.loc[:,'date'].dt.year


# In[181]:


top_boller_each_season=boller.groupby(['season','bowler'])['is_wicket'].sum().reset_index().sort_values(by=['season','is_wicket'],ascending=[True,False])


# In[182]:


top_boller_each_season


# In[183]:


top_boller_each_season['Rank']=top_boller_each_season.groupby('season')['is_wicket'].rank(method='first',ascending=False)


# In[184]:


top_boller_each_season


# In[185]:


top_boller_each_season.loc[top_boller_each_season['Rank']==1.0]


# In[186]:


best_boller_each_season=pd.merge(avg_wickets,top_boller_each_season,left_on=['bowler','season'],right_on=['bowler','season'])


# In[187]:


best_boller_each_season


# In[188]:


unique_boller = best_boller_each_season['bowler'].unique()
num_bollers = len(unique_boller)
fig, axes = plt.subplots(nrows=num_bollers, ncols=1, figsize=(10, num_bollers * 4), sharex=True)

for ax, boller_ in zip(axes, unique_boller):
    boller_data = best_boller_each_season[best_boller_each_season['bowler'] == boller_]
    for team in boller_data['bowling_team'].unique():
        team_data = boller_data[boller_data['bowling_team'] == team]
        ax.plot(
            team_data['season'],
            team_data['Rank'],
            marker='o',
            label=f"{team}"
        )
    ax.set_title(f"Ranking of {boller_} Over Years", fontsize=14)
    ax.set_ylabel("Total Wickets", fontsize=12)
    ax.grid(True)
    ax.legend(title="Bowling Team", fontsize=12)

plt.xlabel("Year", fontsize=14)
plt.tight_layout()

plt.show()


# In[189]:


best_boller


# In[190]:


avg_economy = best_boller.groupby(['bowler', 'year','bowling_team'])['economy'].mean().reset_index()
unique_bollers = avg_economy['bowler'].unique()
num_bollers = len(unique_bollers)

fig, axes = plt.subplots(nrows=num_bollers, ncols=1, figsize=(10, num_bollers * 4), sharex=True)


for ax, boller_ in zip(axes, unique_bollers):
    boller_data = avg_economy[avg_economy['bowler'] == boller_]
    for team in boller_data['bowling_team'].unique():
        team_data = boller_data[boller_data['bowling_team'] == team]
        ax.plot(
            team_data['year'],
            team_data['economy'],
            marker='o',
            label=f"{team}"
        )
    ax.set_title(f"Average Economy of {boller_} Over Years", fontsize=14)
    ax.set_ylabel("Average Economy", fontsize=12)
    ax.grid(True)
    ax.legend(title="Bowling Team", fontsize=12)

plt.xlabel("Year", fontsize=14)
plt.tight_layout()

plt.show()


# In[191]:


deliveries.loc[deliveries.batter.isin(best_batsman.batter.unique())].groupby(['batter','overs_group'],as_index=False).agg({'batsman_runs':'sum','is_wicket':'sum','ball':'count'})


# In[192]:


deliveries.loc[deliveries.bowler.isin(best_boller.bowler.unique())].groupby(['bowler','overs_group'],as_index=False).agg({'batsman_runs':'sum','is_wicket':'sum','ball':'count'})


# In[193]:


batter_auction = (
    deliveries.groupby(['overs_group', 'batter'], as_index=False)
    .agg({'batsman_runs': 'sum', 'is_wicket': 'sum', 'ball': 'count'})
    .sort_values(['overs_group', 'batsman_runs'], ascending=[False, False])
    .groupby('overs_group')
    .head(5)
)


# In[242]:


batter_auction['strike_rate']=batter_auction['batsman_runs']/batter_auction['ball']
batter_auction


# In[243]:


import random as rd
abd_pop=deliveries.loc[(deliveries.batter=="AB DE VILLIERS")&(deliveries.overs_group=="Death")]

other_pop=deliveries.loc[(deliveries.batter!="AB DE VILLIERS")&(deliveries.overs_group=="Death")]


# In[246]:


from scipy.stats import ttest_ind

# Per-ball strike rate for V. Kohli
abd_strike_rate = vk_pop['batsman_runs']

# Per-ball strike rates for other batters
other_strike_rate = other_pop['batsman_runs']

# Perform t-test
t_stat, p_value = ttest_ind(abd_strike_rate, other_strike_rate, equal_var=False)

# Print results
print(f"T-Test Results:\n"
      f"t-statistic: {t_stat:.2f}, p-value: {p_value:.4f}")

# Interpret the result
if p_value < 0.05:
    print("Reject the null hypothesis: AB DE VILLIERS's strike rate is significantly better.")
else:
    print("Fail to reject the null hypothesis: No significant difference in strike rates.")


# In[247]:


import numpy as np
import scipy.stats as stats

# Calculate means and standard deviations for both groups
abd_mean = abd_strike_rate.mean()
other_mean = other_strike_rate.mean()
abd_std = abd_strike_rate.std(ddof=1)
other_std = other_strike_rate.std(ddof=1)

# Number of samples
n_abd = len(abd_strike_rate)
n_other = len(other_strike_rate)

# Calculate the standard error of the difference in means
se = np.sqrt((abd_std**2 / n_abd) + (other_std**2 / n_other))

# Calculate the 95% confidence interval for the difference in means
confidence_level = 0.95
alpha = 1 - confidence_level
df = min(n_abd - 1, n_other - 1)  # degrees of freedom for Welch's t-test

# t-critical value
t_critical = stats.t.ppf(1 - alpha / 2, df)

# Margin of error
margin_of_error = t_critical * se

# Confidence interval for the difference in means
mean_difference = abd_mean - other_mean
ci_lower = mean_difference - margin_of_error
ci_upper = mean_difference + margin_of_error

print(f"Confidence Interval for the Difference in Strike Rates: ({ci_lower:.2f}, {ci_upper:.2f})")

