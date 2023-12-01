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

```python
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

df_twitter = pd.read_csv('df_twitter.csv')
```
```python
df_twitter.describe()

```
Remove the duplicates from the data

```python
# Removing the duplicates

df_twitter.drop_duplicates(subset='full_text',inplace=True)

```

```python
df_twitter.shape
```
(844, 20)

### Importing the stop words

```python

# Import stopwords and define more additional stopwords that you want to include
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('stopwords')

from nltk.corpus import stopwords

additional = ['rt','rts','retweet','fcpd','officer','today','one', 'day', 'fairfax','thanks', 'amp','officers']

swords = set().union(stopwords.words('english'),additional)

```
```python
print(swords)
```
{'an', 'not', 'day', "that'll", 'are', 'no', 'until', 'theirs', "you'll", 'against', "you'd", 'when', 'it', 'your', "couldn't", 'can', 'own', 'we', 'nor', 'below', 'who', 'while', 'very', 'has', 'of', 'i', 'why', 'each', 'what', 'haven', 'they', 'isn', 'rts', 'him', 'doesn', 'the', 'hadn', 'or', 'officer', 'yours', "won't", 'their', 'once', "mustn't", "wasn't", "weren't", 'wouldn', 'off', 'mustn', 'herself', 'them', 'had', 'into', "you've", 'am', 's', "hadn't", 'during', 'and', 'whom', 'aren', 'today', 't', 'wasn', 'having', "needn't", 'fcpd', 'couldn', 'a', "shan't", 'yourselves', 'will', 'above', 'ourselves', 'd', 'hasn', 'where', 'she', "aren't", "hasn't", 'that', 'mightn', 'one', 'myself', "she's", 'too', 'did', 're', 'more', 'is', 'out', 'didn', "don't", 'again', 'here', 've', "wouldn't", 'how', 'most', 'those', "mightn't", 'been', 'to', 'weren', 'from', 'officers', 'then', 'now', 'on', "haven't", 'under', 'm', 'just', 'by', 'so', 'being', 'such', 'rt', 'thanks', 'be', 'have', 'both', 'there', 'same', 'through', "shouldn't", 'me', 'as', 'these', 'than', 'only', 'should', 'retweet', "you're", 'was', 'himself', 'this', 'any', 'after', 'does', 'before', 'ma', 'don', 'her', 'do', 'but', 'over', 'shouldn', 'its', 'up', 'my', 'because', 'between', 'amp', 'hers', 'won', 'shan', 'further', 'fairfax', 'y', 'll', 'all', "it's", 'our', "doesn't", 'few', 'themselves', 'ours', 'if', "should've", "isn't", 'you', 'which', 'about', 'yourself', 'ain', 'needn', 'at', 'other', 'some', 'his', 'itself', "didn't", 'were', 'he', 'with', 'doing', 'for', 'down', 'in', 'o'}

```python
df_twitter['processed_text'] = df_twitter['full_text'].str.lower()\
          .str.replace('(@[a-z0-9]+)\w+',' ')\
          .str.replace('(http\S+)', ' ')\
          .str.replace('([^0-9a-z \t])',' ')\
          .str.replace(' +',' ')\
          .apply(lambda x: [i for i in x.split() if not i in swords])

# Removing more unncessary similar words

from nltk.stem import PorterStemmer

ps = PorterStemmer()
df_twitter['stemmed'] = df_twitter['processed_text'].apply(lambda x: [ps.stem(i) for i in x if i != ''])
df_twitter['processed_text']
```
#### Creating a Word Cloud

```python
from wordcloud import WordCloud

# We need strings for wordcloud. Since the processed_text column is a list we'll convert it into strings

text = ' '.join(df_twitter['processed_text'].apply(lambda x: ' '.join(x)))

# Define wordcloud

wordcloud = WordCloud(width=800, height=400).generate(text)

# Plooting the words

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

```
<img src="https://github.com/Deepti1206/Twitter_Sentiment_Analysis/blob/main/Images/Screenshot%202023-11-30%20at%2010.44.36%20PM.png" width="700" height="450">

It can be said that the FCPD tweets contains emergency updates, crime news and general information regarding law as observed through the commmon words in a wordcloud

#### Sentiment Analysis

We will perform a very bbasic sentiment analysis

```python
import nltk.sentiment.vader as vd
from nltk import download
download('vader_lexicon')

sia = vd.SentimentIntensityAnalyzer()

nltk.download('punkt')
from nltk.tokenize import word_tokenize

df_twitter['sentiment_score'] = df_twitter['processed_text'].apply(lambda x: sum([ sia.polarity_scores(i)['compound'] 
                                                                  for i in word_tokenize( ' '.join(x) )]) )

df_twitter[['processed_text','sentiment_score']].head(n=50)

```python
df_twitter['sentiment_score'].apply(lambda x: round(x,)).value_counts()
```
 0    422   
 1    183   
-1    175   
 2     32   
-2     18   
 3     13   
-4      1      
Name: sentiment_score, dtype: int64

We can classify the Fairfax police updates into 3 categories:

Positive Tweets : Sentiment score > 0 are positive  
Neutral Tweets : Sentiment score = 0 are neutral  
Negative Tweets : Sentiment score < 0 are negative  

```python
# Define a function to calculate sentiment score

def calculate_sentiment_score(sentiment):
    if sentiment < 0:
        return 'negative'
    elif sentiment > 0:
        return 'positive'
    else:
        return 'neutral'

# Create a new column for category tags
df_twitter['category'] = ''

# Calculate sentiment scores and assign category tags
for index, row in df_twitter.iterrows():
    sentiment_score = row['sentiment_score']
    category = calculate_sentiment_score(sentiment_score)
    df_twitter.at[index, 'category'] = category

# Print the counts of tweets in each category
print(df_twitter['category'].value_counts())
```
positive    373    
negative    347    
neutral     124    
Name: category, dtype: int64

```python
df_senti = df_twitter[['processed_text','sentiment_score','category']]

category_counts = df_senti['category'].value_counts()

plt.figure(figsize=(10, 8))
patches, texts, autotexts = plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90, 
                                   colors=['lightblue', 'lightgreen', 'lightcoral'], wedgeprops=dict(width=0.3))

# Adjusting the position of labels
plt.legend(patches, category_counts.index, loc='center left', bbox_to_anchor=(1, 0.5))

# Displaying the percentage labels outside the pie chart
for autotext in autotexts:
    autotext.set_position((1.2 * autotext.get_position()[0], autotext.get_position()[1]))

plt.title('Distribution of Categories')
plt.show()
```
<img src="https://github.com/Deepti1206/Twitter_Sentiment_Analysis/blob/main/Images/Screenshot%202023-11-30%20at%2010.14.58%20PM.png" width="600" height="450">

As per the above sentiment analysis, it can be said that the 44.2% of the time, the Fairfax Poilce tweets have positive sentiments but on the other side 41% has the negative sentiments. For negative sentiments, there could be possible emergency updates that they are posting on twitter.

