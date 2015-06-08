from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import re
import csv
import time

start_time = time.clock()

raw_data = []
doc = None
with open('fixtures/199801_clear.txt', 'r') as f:
    while 1:
        if doc is None:
            doc = []
        line = f.readline().decode('gbk')
        if not line:
            break
        if line[0] == '1':
            word_list = re.split('/\w+\s', line)[1:-1]
            doc.extend(word_list)
        else:
            if not doc:
                continue
            raw_data.append(" ".join(doc))
            doc = None

vectorizer = TfidfVectorizer()

print time.ctime() + ' --> fit transform '
vec_matrix = vectorizer.fit_transform(raw_data).toarray()
print time.ctime() + ' --> calc distance'

start_calc_time = time.clock()
dist = pairwise_distances(vec_matrix, metric='cosine')
end_calc_time = time.clock()
print "similary calculation time --> " + str(end_calc_time - start_calc_time)

print time.ctime() + ' --> start write'
writer = csv.writer(file('output.csv', 'w'))
for row in dist:
    writer.writerow(row)

end_time = time.clock()
print "total time --> " + str(end_time - start_time)
