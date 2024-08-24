# Machine-Learning---Sentiment-Analysis-Project
Build a predictive model to understand customer behaviour 

# SENTIMENT ANALYSIS 

#Import Packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

import nltk
import re

import string

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sentiments=SentimentIntensityAnalyzer()

from wordcloud import WordCloud , STOPWORDS , ImageColorGenerator

nltk.download('stopwords')
from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))


#To extract reviews from multiple pages
all_reviews = []
for page_num in range(1, 10):

    url_source = 'https://www.airlinequality.com/airline-reviews/british-airways/?sortby=post_date%3ADesc&pagesize=100&page={page_num}'
    url=url_source.format(page_num=page_num)
    r=requests.get(url)

#extract data using BeautifulSoup library
soup=BeautifulSoup(r.content, 'lxml')

reviews = soup.find_all('h2', {'class': "text_header"})
div_reviews = soup.find_all('div', {'class': "text_content"})
                                    
#To add the reviews of the next page one after the other
review2 = ' '
for review in reviews:
    review2 += review.text.strip() + '\n'
                                    
for div_review in div_reviews:
    review2 + div_review.text.strip() + '\n'
all_reviews.append(review2)
                                    
print(all_reviews)


#Extract into a csv file
for review in all_reviews:
    with open('BA_Review.csv', 'a', encoding = 'utf-8') as f:
        f.write(review + '\n')

#Load input dataframe
df = pd.read_csv(r"C:\\Users\\USER\\Documents\\Jupyter Workings\\BA_Review.csv")
df

print(df.head(10))


#check for null values
print(df.isnull().sum())

stemmer=nltk.SnowballStemmer("english")


#Apply regex to remove unwanted characters
#cleaned_text = re.sub(pattern, " ", str(REVIEW))

#Clean Datatset
def clean(text):
	text=str(text).lower()
	text=re.sub('\[.*?\]', '',text)
	text=re.sub('https?://\S+|WWW\.\S+','',text)
	text=re.sub('<.*?>+','',text)
	text=re.sub('\n','',text)
	text=re.sub('\W*\d\W*','',text)
	#ext=re.sub('[%%S]' %% re.escape(string.punctuation),'',text)
	
	text=[word for word in text.split(' ')]
	text=" ".join(text)
	text=[stemmer.stem(word) for word in text.split(' ')]
	text=" ".join(text)
	return text

df["REVIEW"]=df["REVIEW"].apply(clean)

df["REVIEW"].apply(clean)


# SENTIMENT INTENSITY ANALYSIS
nltk.download('vader_lexicon')
sentiments=SentimentIntensityAnalyzer()

df['Positive']=[sentiments.polarity_scores(i)["pos"] for i in df["REVIEW"]]
df['Negative']=[sentiments.polarity_scores(i)["neg"] for i in df["REVIEW"]]
df['Neutral']=[sentiments.polarity_scores(i)["neu"] for i in df["REVIEW"]]

df=df[["REVIEW","Positive","Negative","Neutral"]]

print(df.head(10))


#Overall Sentiment Score
#Determine the Average Polarity Score
x=sum(df["Positive"])
y=sum(df["Negative"])
z=sum(df["Neutral"])

def sentiment_score(a,b,c):
    if (a>b) and (a>c):
        print("Positive")
    elif (b>a) and (b>c):
        print("Negative")
    else:
        print("Neutral")

print("Positive:", x)
print("Negative:", y)
print("Neutral:", z)

sentiment_score(x,y,z)


#Create a bar chart
plt.figure(figsize=(7,5))
plt.title('Sentiment Score')
plt.bar(['positve','Negative','Neutral'], [x,y,z])
plt.show

# Plotting Word Cloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#df=df[["REVIEW","Positive","Negative","Neutral"]]

# Change reviews to strings
Reviews = ' '.join(review for review in df["REVIEW"])
type(Reviews)

#Create a Word Cloud 
Reviews_wordcloud = WordCloud(width=600, height=400, background_color='White').generate(Reviews)
plt.figure(figsize=(15,10))
plt.imshow(Reviews_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Create a Pie Chart
from pylab import *

# make a square figure and axes
figure(1, figsize=(6,6))
ax = axes([0.1, 0.1, 0.8, 0.8])

# The slices will be ordered and plotted counter-clockwise.
labels = 'Positive', 'Negative', 'Neutral'
fracs = [110.61, 67.24, 422.15]
explode=(0,  0, 0.07)

pie(fracs, explode=explode, labels=labels,
                autopct='%1.1f%%', shadow=True, startangle=90)
                # The default startangle is 0, which would start
                # the Neutral slice on the x-axis.  With startangle=90,
                # everything is rotated counter-clockwise by 90 degrees,
                # so the plotting starts on the positive y-axis.

title('Sentiment Analysis', bbox={'facecolor':'0.9', 'pad':7})

show()




