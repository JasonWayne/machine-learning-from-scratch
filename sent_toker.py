from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
import csv

toker = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)

writer = csv.writer(file('output.txt', 'w'))
with open('input.txt') as f:
	count = 0
	while 1:
		count += 1
		line = f.readline()
		if not line:
			break
		try:
			tokens = toker.tokenize(line)
		except UnicodeDecodeError:
			print count
		# remove stopwords
		tokens = [w.lower() for w in tokens if w not in stopwords.words('english')]
		# remove numbers
		tokens = [w for w in tokens if w.isalpha()]
		writer.writerow(tokens)
	print count
