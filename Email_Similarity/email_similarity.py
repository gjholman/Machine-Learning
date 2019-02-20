from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


'''
We want to be able to label emails based their content. 
In this case, we're looking to differentiate baseball and hockey
'''

# Grab training set, indicated by subset='train'
train_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset = 'train', shuffle=True, random_state=108)

# Grab testing set
test_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset = 'test', shuffle=True, random_state=108)

# Create Count Vectorizer so we can tag the words and categorize them
counter = CountVectorizer()
counter.fit(test_emails.data + train_emails.data)

# We can now make a list of counts of our words in our training and testing set
train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)

# Time for Naive Bayes Classification
# Fit takes in training set (train_counts), and the labels assosiated (train_emails.target)
classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)

# Finally, let's see how accurate our model is
print(classifier.score(test_counts, test_emails.target))

'''
This model is 97.24% accurate
That's pretty good! Now we can play around with the other categories:

'alt.atheism'
'comp.graphics'
'comp.os.ms-windows.misc'
'comp.sys.ibm.pc.hardware'
'comp.sys.mac.hardware'
'comp.windows.x'
'misc.forsale'
'rec.autos'
'rec.motorcycles'
'rec.sport.baseball'
'rec.sport.hockey'
'sci.crypt'
'sci.electronics'
'sci.med'
'sci.space'
'soc.religion.christian'
'talk.politics.guns'
'talk.politics.mideast'
'talk.politics.misc'
'talk.religion.misc'

'''
