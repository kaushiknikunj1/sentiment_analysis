#installing basic libraries etc.

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
!pip install numpy pandas seaborn matplotlib textblob nltk wordcloud sklearn corpus vaderSentiment



#upload file with comments dataframe
df = pd.read_csv('/dbfs/FileStore/shared_uploads/sentiment_analysis_df.csv')

print(df)

final_df = df[['translated_text']].dropna()

print(final_df)

#Clustering Model

    text = final_df['translated_text'].astype(str)

    vectorizer = TfidfVectorizer(stop_words={'english'})
    X = vectorizer.fit_transform(text)

    
    Sum_of_squared_distances = []
    K = range(2,10)
    for k in K:
       km = KMeans(n_clusters=k, max_iter=200, n_init=10)
       km = km.fit(X)
       Sum_of_squared_distances.append(km.inertia_)
    
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    #print('How many clusters do you want to use?')
    
    true_k = 6
    
    #true_k = int(input())
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
    model.fit(X)

    labels=model.labels_
    clusters=pd.DataFrame(list(zip(text,labels)),columns=['text','cluster'])
    print(clusters.sort_values(by=['cluster']))

#     for i in range(true_k):
#         print(clusters[clusters['cluster'] == i])


##Vader Model

from nltk.stem import WordNetLemmatizer

import pandas as pd
import re

#df = pd.read_csv('/dbfs/FileStore/shared_uploads/sanjay/sentiment_analysis_df.csv')

df = pd.read_csv('/dbfs/FileStore/shared_uploads/sanjay/dhantara_video_comments_v2___Sheet1.csv')
#print(df)

#final_df = df[['plain_text']]


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# function to print sentiments
def sentiment_scores(sentence):

	# Create a SentimentIntensityAnalyzer object.
	sid_obj = SentimentIntensityAnalyzer()

	polarity_scores method of SentimentIntensityAnalyzer
	object gives a sentiment dictionary.
	which contains pos, neg, neu, and compound scores.
	sentiment_dict = sid_obj.polarity_scores(sentence)
	
	print("Overall sentiment dictionary is : ", sentiment_dict)
	print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
	print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
	print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")

# 	print("Sentence Overall Rated As", end = " ")
    
# 	# decide sentiment as positive, negative and neutral
	return 1.0*sentiment_dict['compound']
#         if sentiment_dict['compound'] >= 0.05 :
#             print("Positive")

#         elif sentiment_dict['compound'] <= - 0.05 :
#             print("Negative")

#         else :
#             print("Neutral")
