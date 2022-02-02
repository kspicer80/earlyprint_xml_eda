from pathlib import Path, PurePath
import glob
from collections import Counter
from lxml import etree
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
nsmap={'tei': 'http://www.tei-c.org/ns/1.0'}

files = glob.glob('datasets/*.xml')
filekeys = [f.split('/')[1].split('.')[0] for f in files]
metadata_files = glob.glob('/Users/spicy.kev/Documents/github/epmetadata')
all_tokenized = []

filename_list = []

test_string = 'datasets/A27177_07.xml'
#first_split = test_string.split('/')[-1].split('.')[0]
#print(first_split)

for f in files:
    parser = etree.XMLParser(collect_ids=False)
    tree = etree.parse(f, parser)
    xml = tree.getroot()

    word_tags = xml.findall('.//{*}w')
    # Grab all regularied forms
    #words = [word.get('reg', word.text).lower() for word in word_tags if word.text != None]
    # Grab all the lemmas
    words = [word.get('lemma', word.text).lower() for word in word_tags if word.text != None]
    all_tokenized.append(words)
all_counted = [Counter(a) for a in all_tokenized]
#print(all_counted)
df = pd.DataFrame(all_counted, index=filekeys).fillna(0)
#print(df.head())

tfidf = TfidfTransformer(norm=None, sublinear_tf=True)
results = tfidf.fit_transform(df)
readable_results = pd.DataFrame(results.toarray(), index=df.index, columns=df.columns)
print(readable_results.T.sort_values(by='A27177_07', ascending=False).head(30))
