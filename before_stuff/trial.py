import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
#plt.show()

tweet = "That still doesn't make up for the Steelers losing to the Raiders but whatever."
wordcloud = WordCloud().generate(tweet)
#print (wordcloud.values())
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('foo.png')
#plt.show() 
