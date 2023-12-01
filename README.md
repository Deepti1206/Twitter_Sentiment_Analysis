# Twitter Sentiment Analysis from Fairfax Police Tweets

<img src="https://github.com/Deepti1206/Twitter_Sentiment_Analysis/blob/main/Images/Screenshot%202023-11-30%20at%2010.44.36%20PM.png" width="700" height="450">

This is a basic analysis of sentiments through the tweets of FCPD on twitter

Sentiment Analysis of Fairfax Police Tweets from Twitter (now X)
In this project, sentiment analysis was conducted on tweets from the Fairfax Police. The goal was to evaluate the sentiment expressed in each tweet.. Following the sentiment analysis, tweets were categorized into three distinct groups: "Important" , "General" and "Non-Important." This categorization provides valuable insights into the nature of sentiments expressed in the tweets, helping to distinguish between tweets of significance and those of lesser importance.

The following steps were performed from data export to sentiment analysis:

Certainly! Here are the numbered points extracted from the last message:

1. Data Export
2. Data Preprocessing
3. Text Preprocessing
4. Sentiment Analysis
5. Categorization
6. Visualization
7. Analysis and Interpretation

*I have skipped the data cleaning part for this project, we will directly start exportimng the cleaned data. You will find in the folder above.*

#### Import the libraries and load the data

```{python}
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

df_twitter = pd.read_csv('df_twitter.csv')
```
```{python r}
df_twitter.describe()

```
Remove the duplicates from the data

```{python w}
# Removing the duplicates

df_twitter.drop_duplicates(subset='full_text',inplace=True)

```

```{python w}
df_twitter.shape
```
(844, 20)

### Importing the stop words

```{python}

# Import stopwords and define more additional stopwords that you want to include
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('stopwords')

from nltk.corpus import stopwords

additional = ['rt','rts','retweet','fcpd','officer','today','one', 'day', 'fairfax','thanks', 'amp','officers']

swords = set().union(stopwords.words('english'),additional)






```


