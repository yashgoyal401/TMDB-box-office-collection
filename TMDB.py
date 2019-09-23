# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 23:15:20 2019

@author: yashgoyal
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(context="notebook", style="darkgrid", palette="deep", font="sans-serif", font_scale=1, color_codes=True)

train  = pd.read_csv("C:\\Users\\yashgoyal\\TMDB_train.csv")
test  = pd.read_csv("C:\\Users\\yashgoyal\\TMDB_test.csv")

train.shape
test.shape
EDA = train.describe(include="all")
train.isnull().sum()
test.isnull().sum()
sns.heatmap(train.isnull())
sns.heatmap(test.isnull())

train.columns
sns.jointplot('budget','revenue',kind="scatter",data=train)
sns.jointplot('revenue','popularity',data=train)
sns.jointplot('runtime','revenue',data=train)

sns.distplot(train.revenue)
y_train = train.revenue
from scipy.special import boxcox1p,inv_boxcox1p,boxcox
y_train=boxcox1p(y_train,0.2)
sns.distplot(y_train)


###### Release_ date
#Since only last two digits of year are provided, this is the correct way of getting the year.
train[['release_month','release_day','release_year']]=train['release_date'].str.split('/',expand=True).replace(np.nan, -1).astype(int)
# Some rows have 4 digits of year instead of 2, that's why I am applying (train['release_year'] < 100) this condition
train.loc[ (train['release_year'] <= 19) & (train['release_year'] < 100), "release_year"] += 2000
train.loc[ (train['release_year'] > 19)  & (train['release_year'] < 100), "release_year"] += 1900

releaseDate = pd.to_datetime(train['release_date']) 
train['release_dayofweek'] = releaseDate.dt.dayofweek
train['release_quarter'] = releaseDate.dt.quarter

plt.figure(figsize=(20,12))
sns.countplot(train.release_year)
plt.xticks(rotation=90)

plt.figure(figsize=(20,12))
sns.countplot(train.release_month)
plt.xticks(rotation=90)

plt.figure(figsize=(20,12))
sns.countplot(train.release_day)
plt.xticks(rotation=90)

plt.figure(figsize=(20,12))
sns.countplot(train.release_dayofweek)
plt.xticks(rotation=90)

plt.figure(figsize=(20,12))
sns.countplot(train['release_dayofweek'])
plt.title("Total movies released on Day Of Week",fontsize=20)
loc, labels = plt.xticks()
loc, labels = loc, ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
plt.xticks(loc, labels,fontsize=20)
plt.show()

plt.figure(figsize=(20,12))
sns.countplot(train.release_quarter)
plt.xticks(rotation=90)

plt.figure(figsize=(22,13))
train.groupby("release_year")["revenue"].mean().plot()
plt.xticks(np.arange(1920,2020,4))
plt.xticks(rotation=90)

plt.figure(figsize=(22,13))
train.groupby("release_month")["revenue"].mean().plot()
plt.xticks(rotation=90)
plt.xticks(np.arange(1,12,1))

plt.figure(figsize=(22,13))
train.groupby("release_quarter")["revenue"].mean().plot()
plt.xticks(rotation=90)
plt.xticks(np.arange(1,4,1))

plt.figure(figsize=(22,13))
train.groupby("release_year")["budget"].mean().plot()
plt.xticks(rotation=90)
plt.xticks(np.arange(1920,2020,4))

plt.figure(figsize=(22,13))
train.groupby("release_year")["runtime"].mean().plot()
plt.xticks(rotation=90)
plt.xticks(np.arange(1920,2020,4))

plt.figure(figsize=(22,13))
train.groupby("release_year")["popularity"].mean().plot()
plt.xticks(rotation=90)
plt.xticks(np.arange(1920,2020,4))

train = train.drop(["release_date"],axis=1)
### Homepage
train.homepage.isnull().sum()
train["has_homepage"] = 1
train.loc[pd.isnull(train["homepage"]),"has_homepage"] = 0
sns.countplot(train.has_homepage)
train = train.drop(["homepage"],axis=1)

## Tagline
train["has_tagline"] = 1
train.loc[pd.isnull(train["tagline"]),"has_tagline"] = 0
sns.countplot(train.has_tagline)
sns.catplot(x="has_tagline",y="revenue",data=train)
train = train.drop(["tagline"],axis=1)

### Status
train.status.unique()
train.status.value_counts()
train["status"] = train.status.replace({"Released":1,"Rumored":0})
sns.catplot(x="status",y="revenue",data=train)

#### Title and original title
### TitleSame=0, TitleDifferent=1
train["IsTitleDifferent"] = 1
train.loc[train['original_title'] == train['title'],"IsTitleDifferent"] = 0
sns.countplot(train.IsTitleDifferent)
sns.catplot(x="IsTitleDifferent",y="revenue",data=train)
train = train.drop(["original_title","title"],axis=1)

#### Genres
genres_count=[]
for i in train['genres']:
    if(not(pd.isnull(i))):
        
        genres_count.append(len(eval(i)))
        
    else:
        genres_count.append(0)
train['num_genres'] = genres_count
train = train.drop(["genres"],axis=1)

#### Belongs to collection
train["has_collection_company"]=1
train.loc[train["belongs_to_collection"].isnull(),"has_collection_company"]=0
sns.countplot(train.has_collection_company)
sns.catplot(x="has_collection_company",y="revenue",data=train)
train = train.drop(["belongs_to_collection"],axis=1)

#### Spoken language 
train.columns
train.spoken_languages.value_counts()

language_count = []
for i in train["spoken_languages"]:
    if(not(pd.isnull(i))):
        language_count.append(len(eval(i)))
    else:
        language_count.append(0) 
train["num_languages"] =  language_count       
train = train.drop(["spoken_languages"],axis=1)    
sns.countplot(train.num_languages)

### Original language
train.original_language.unique()
train["original_language"] = train["original_language"].replace({'en':1, 'hi':2, 'ko':3, 'sr':4, 'fr':5, 'it':6, 'nl':7, 'zh':8, 'es':9, 'cs':10, 'ta':11,
       'cn':12, 'ru':13, 'tr':14, 'ja':15, 'fa':16, 'sv':17, 'de':18, 'te':19, 'pt':20, 'mr':21, 'da':22,
       'fi':23, 'el':24, 'ur':25, 'he':26, 'no':27, 'ar':28, 'nb':29, 'ro':30, 'vi':31, 'pl':32, 'hu':33,
       'ml':34, 'bn':35, 'id':36})
train = train.drop(["imdb_id","overview","poster_path"],axis=1)

#### Production companies
train.production_companies.unique()
num_productionCompanies = []
for i in train["production_companies"]:
      if(not(pd.isnull(i))):
          num_productionCompanies.append(len(eval(i)))
      else:
          num_productionCompanies.append(0)
train["num_ProdComp"] = num_productionCompanies
train = train.drop(["production_companies"],axis=1)

#### Production countries
num_productionCountries = []
for i in train["production_countries"]:
      if(not(pd.isnull(i))):
          num_productionCountries.append(len(eval(i)))
      else:
          num_productionCountries.append(0)
train["num_ProdCountries"] = num_productionCountries
train = train.drop(["production_countries"],axis=1)

#### Keywords
num_Keywords = []
for i in train["Keywords"]:
      if(not(pd.isnull(i))):
          num_Keywords.append(len(eval(i)))
      else:
          num_Keywords.append(0)
train["Num_KEYWORDS"] = num_Keywords
train = train.drop(["Keywords"],axis=1)


### cast
num_Cast = []
for i in train["cast"]:
      if(not(pd.isnull(i))):
          num_Cast.append(len(eval(i)))
      else:
          num_Cast.append(0)
train["Num_Cast"] = num_Cast
train = train.drop(["cast"],axis=1)

### Crew
num_Crew = []
for i in train["crew"]:
      if(not(pd.isnull(i))):
          num_Crew.append(len(eval(i)))
      else:
          num_Crew.append(0)
train["Num_Crew"] = num_Crew
train = train.drop(["crew"],axis=1)

X_train = train.drop(["revenue","id"],axis=1)
X_train.isnull().sum()
sns.heatmap(X_train.isnull())
X_train["runtime"] = X_train["runtime"].fillna(X_train["runtime"].mean())

######### Test data ##########
test.columns
test  = test.drop(["overview","poster_path","imdb_id"],axis=1)

#Since only last two digits of year are provided, this is the correct way of getting the year.
test[['release_month','release_day','release_year']]=test['release_date'].str.split('/',expand=True).replace(np.nan, -1).astype(int)
# Some rows have 4 digits of year instead of 2, that's why I am applying (train['release_year'] < 100) this condition
test.loc[ (test['release_year'] <= 19) & (test['release_year'] < 100), "release_year"] += 2000
test.loc[ (test['release_year'] > 19)  & (test['release_year'] < 100), "release_year"] += 1900

releaseDate = pd.to_datetime(test['release_date']) 
test['release_dayofweek'] = releaseDate.dt.dayofweek
test['release_quarter'] = releaseDate.dt.quarter
test = test.drop(["release_date"],axis=1)


### Homepage
test.homepage.isnull().sum()
test["has_homepage"] = 1
test.loc[pd.isnull(test["homepage"]),"has_homepage"] = 0
sns.countplot(test.has_homepage)
test = test.drop(["homepage"],axis=1)

## Tagline
test["has_tagline"] = 1
test.loc[pd.isnull(test["tagline"]),"has_tagline"] = 0
sns.countplot(test.has_tagline)
test = test.drop(["tagline"],axis=1)

### Status
test.status.unique()
test.status.value_counts()
test["status"] = test.status.replace({"Released":1,"Rumored":0,"Post Production":0})
sns.countplot(test.status)

#### Title and original title
### TitleSame=0, TitleDifferent=1
test["IsTitleDifferent"] = 1
test.loc[test['original_title'] == test['title'],"IsTitleDifferent"] = 0
sns.countplot(test.IsTitleDifferent)
test= test.drop(["original_title","title"],axis=1)

#### Genres
genres_count=[]
for i in test['genres']:
    if(not(pd.isnull(i))):
        
        genres_count.append(len(eval(i)))
        
    else:
        genres_count.append(0)
test['num_genres'] = genres_count
test = test.drop(["genres"],axis=1)   


#### Belongs to collection
test["has_collection_company"]=1
test.loc[test["belongs_to_collection"].isnull(),"has_collection_company"]=0
sns.countplot(test.has_collection_company)
test = test.drop(["belongs_to_collection"],axis=1)

#### Spoken language 
test.columns
test.spoken_languages.value_counts()

language_count = []
for i in test["spoken_languages"]:
    if(not(pd.isnull(i))):
        language_count.append(len(eval(i)))
    else:
        language_count.append(0) 
test["num_languages"] =  language_count       
test = test.drop(["spoken_languages"],axis=1)    
sns.countplot(test.num_languages)

### Original language
test.original_language.unique()
test.original_language.value_counts()
test["original_language"] = test["original_language"].replace({'en':1, 'hi':2, 'ko':3, 'sr':4, 'fr':5, 'it':6, 'nl':7, 'zh':8, 'es':9, 'cs':10, 'ta':11,
       'cn':12, 'ru':13, 'tr':14, 'ja':15, 'fa':16, 'sv':17, 'de':18, 'te':19, 'pt':20, 'mr':21, 'da':22,
       'fi':23, 'el':24, 'ur':25, 'he':26, 'no':27, 'ar':28, 'nb':29, 'ro':30, 'vi':31, 'pl':32, 'hu':33,
       'ml':34, 'bn':35, 'id':36,'ka':37,'th':38,'ca':39,'bm':40,'af':41,'xx':42,'kn':43,'is':44})


#### Production companies
test.production_companies.unique()
num_productionCompanies = []
for i in test["production_companies"]:
      if(not(pd.isnull(i))):
          num_productionCompanies.append(len(eval(i)))
      else:
          num_productionCompanies.append(0)
test["num_ProdComp"] = num_productionCompanies
test = test.drop(["production_companies"],axis=1)

#### Production countries
num_productionCountries = []
for i in test["production_countries"]:
      if(not(pd.isnull(i))):
          num_productionCountries.append(len(eval(i)))
      else:
          num_productionCountries.append(0)
test["num_ProdCountries"] = num_productionCountries
test = test.drop(["production_countries"],axis=1)

#### Keywords
num_Keywords = []
for i in test["Keywords"]:
      if(not(pd.isnull(i))):
          num_Keywords.append(len(eval(i)))
      else:
          num_Keywords.append(0)
test["Num_KEYWORDS"] = num_Keywords
test = test.drop(["Keywords"],axis=1)

### cast
num_Cast = []
for i in test["cast"]:
      if(not(pd.isnull(i))):
          num_Cast.append(len(eval(i)))
      else:
          num_Cast.append(0)
test["Num_Cast"] = num_Cast
test = test.drop(["cast"],axis=1)

### Crew
num_Crew = []
for i in test["crew"]:
      if(not(pd.isnull(i))):
          num_Crew.append(len(eval(i)))
      else:
          num_Crew.append(0)
test["Num_Crew"] = num_Crew
test = test.drop(["crew"],axis=1)


test.isnull().sum() 
test["runtime"] = test["runtime"].fillna(test["runtime"].mean())
test["status" ] = test["status" ].fillna(test["status" ].mode()[0]) 
test["release_dayofweek" ] = test["release_dayofweek" ].fillna(test["release_dayofweek" ].mode()[0]) 
test["release_quarter" ] = test["release_quarter" ].fillna(test["release_quarter" ].mode()[0]) 
sns.heatmap(test.isnull())

X_test = test.drop(["id"],axis=1)

X_train.columns
X_test.columns

###### Building model
import xgboost
from sklearn.metrics import accuracy_score
predictor = xgboost.XGBRegressor()
predictor.fit(X_train,y_train)
pred_train = predictor.predict(X_train)
pred_test = predictor.predict(X_test)

pred_test_original = inv_boxcox1p(pred_test, 0.2)

PP = pd.concat([test.id],axis=1)
PP["revenue"] = pred_test_original
PP.head()

PP.to_csv("TMDB1stTry.csv",index=False)

