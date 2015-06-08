# coding=utf-8
import numpy as np
import time
import re
from collections import defaultdict
import math
import csv


start_time = time.clock()

doc = None
# the list of all the docs tf
# one dictionary for each doc
tf_list = []

word_set_list = []

# records how many times each word occurs in different docs
df_dict = defaultdict(lambda: 0)

norm_dict = defaultdict(lambda: 0)

with open('fixtures/199801_clear.txt', 'r') as f:
	while True:
		if doc is None:
			doc = []
		line = f.readline().decode('gbk')
		if not line:
			break
		if line[0] == '1':
			# split by the tag and space
			# \w refers to the tag, \s the trailing space
			# :-1 removes the last element, which is always '\r\n'
			word_list = re.split('/\w+\s', line)[1:-1]
			doc.extend(word_list)
		else:
			# consequtive empty line may result in an empty doc here
			# so just check and continue if true
			if not doc:
				continue
			d = defaultdict(lambda: 0)
			base_frequency = 1.0 / len(doc)
			for word in doc:
				d[word] += base_frequency
			tf_list.append(d)
			word_set_list.append(set(doc))
			for word in d.keys():
				df_dict[word] += 1
			doc = None

total_docs = len(tf_list)
result = np.empty((total_docs, total_docs))
count = 0

calc_start = time.clock()

# turn tf value into tf-idf value
for d in tf_list:
	for word in d:
		d[word] = d.get(word) * \
			math.log(total_docs / df_dict[word] + 1, 2)

for i in range(total_docs):
	for j in range(i, total_docs):
		count += 1
		if count % 1000 == 0:
			print count
		if i == j:
			result[i][j] = 1
		else:
			word_set = set.union(word_set_list[i], word_set_list[j])
			vec_i = np.zeros(len(word_set))
			vec_j = np.zeros(len(word_set))
			for index, word in enumerate(word_set):
				tf_idf = tf_list[i].get(word)
				if tf_idf:
					vec_i[index] = tf_idf
			for index, word in enumerate(word_set):
				tf_idf = tf_list[j].get(word)
				if tf_idf:
					vec_j[index] = tf_idf
			# dot = sum(map(lambda x, y: x * y, vec_i, vec_j))
			dot = np.dot(vec_i, vec_j)
			norm_i = norm_dict.get(i)
			norm_j = norm_dict.get(j)
			if not norm_i:
				# norm_i = sum(map(lambda x: x ** 2, vec_i))
				norm_i = np.dot(vec_i, vec_i)
				norm_i = math.sqrt(norm_i)
				norm_dict[i] = norm_i
			if not norm_j:
				# norm_j = sum(map(lambda x: x ** 2, vec_j))
				norm_j = np.dot(vec_j, vec_j)
				norm_j = math.sqrt(norm_j)
				norm_dict[j] = norm_j
			result[i][j] = dot / (norm_i * norm_j)
calc_end = time.clock()
print "similary calculation time --> " + str(calc_end - calc_start)

writer = csv.writer(file('output.txt', 'w'))
for row in result:
	writer.writerow(row)

end_time = time.clock()


print "total time --> " + str(end_time - start_time)
