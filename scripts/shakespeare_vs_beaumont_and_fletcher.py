import numpy as np
import glob
import collections
from lxml import etree
import scipy.spatial.distance as dist
import itertools
import matplotlib.pyplot as plt
from adjustText import adjust_text
nsmap={'tei': 'http://www.tei-c.org/ns/1.0'}

#mac_files_path = glob.glob("/Users/spicy.kev/Desktop/plays/*.xml")
shakespeare_plays = glob.glob(r"C:\Users\KSpicer\Desktop\plays\*.xml")
beaumont_and_fletcher_plays = glob.glob(r"C:\Users\KSpicer\Documents\GitHub\earlyprint_xml_eda\earlyprint_xml_eda\datasets\*.xml")
combined_files = shakespeare_plays + beaumont_and_fletcher_plays
#print(len(beaumont_and_fletcher_plays))
#print(len(combined_files))

def extract_vocabulary(tokenized_corpus, min_count=1, max_count=float('inf')):
    vocabulary = collections.Counter()
    for document in tokenized_corpus:
        vocabulary.update(document)
    vocabulary = {
        word for word, count in vocabulary.items()
        if count >= min_count and count <= max_count
    }
    return sorted(vocabulary)

def corpus2tdm(tokenized_corpus, vocabulary):
    document_term_matrix = []
    for document in tokenized_corpus:
        document_counts = collections.Counter(document)
        row = [document_counts[word] for word in vocabulary]
        document_term_matrix.append(row)
    return document_term_matrix

subgenres = ['Comedies', 'Tragedies', 'Histories', 'Late Romances']

comedies = [
    "All’s Well That Ends Well",
    "As You Like It",
    "The Comedy of Errors",
    "Love’s Labor’s Lost",
    "Measure for Measure",
    "The Merchant of Venice",
    "The Merry Wives of Windsor",
    "A Midsummer Night’s Dream",
    "Much Ado About Nothing",
    "The Taming of the Shrew",
    "Twelfth Night",
    "Two Gentlemen of Verona"
    ]
histories = [
    'Henry IV, Part I',
    'Henry IV, Part II',
    'Henry V',
    'Henry VI, Part 1',
    'Henry VI, Part 2',
    'Henry VI, Part 3',
    'Henry VIII',
    'King John',
    'Richard II',
    'Richard III'
    ]
tragedies = [
    'Antony and Cleopatra',
    'Coriolanus',
    'Hamlet',
    'Julius Caesar',
    'King Lear',
    'Macbeth',
    'Othello',
    'Romeo and Juliet',
    'Timon of Athens',
    'Titus Andronicus',
    'Troilus and Cressida'
    ]
late_romances = [
    'Pericles',
    "The Winter's Tale",
    "The Tempest",
    'Cymbeline'
    ]
    
b_and_f_plays = [
    'Thierry and Theodoret',
    "Cupid's Revenge",
    'A King and No King',
    "The Maid's Tragedy",
    'Philaster or Love Lies a-Bleeding',
    'The Scornful Lady',
    'The Woman Hater',
    'The Noble Gentleman',
    "The Beggar's Bush",
    'The Coxcomb',
    "Love's Cure, or The Martial Maid",
    "Love's Pilgrimage"
    ]
    
# Create empty lists to put all our (meta)data into
plays, titles, genres = [], [], []

# Then you can loop through the files
for f in combined_files:
    parser = etree.XMLParser(collect_ids=False) # Create a parse object that skips XML IDs (in this case they just slow things down)
    tree = etree.parse(f, parser) # Parse each file into an XML tree
    xml = tree.getroot() # Get the XML from that tree

    # Now we can use lxml to find all the w tags
    #word_tags = xml.findall(".//{*}w")
    word_tags = xml.findall(".//tei:w", namespaces=nsmap)
    #title = xml.find(".//tei:titleStmt//tei:title", namespaces=nsmap).text
    title = xml.find(".//tei:teiHeader//tei:fileDesc//tei:titleStmt//tei:title", namespaces=nsmap).text
    #title = tree.find(".//tei:titleStmt//tei:title", namespaces=nsmap).text
    if title in comedies:
        genres.append('Comedies')
        titles.append(title)
        words = [word.get('reg', word.text).lower() for word in word_tags if word.text != None]
        plays.append(words)
    elif title in histories:
        genres.append('Histories')
        titles.append(title)
        words = [word.get('reg', word.text).lower() for word in word_tags if word.text != None]
        plays.append(words)
    elif title in tragedies:
        genres.append('Tragedies')
        titles.append(title)
        words = [word.get('reg', word.text).lower() for word in word_tags if word.text != None]
        plays.append(words)
    elif title in late_romances:
        genres.append('Late Romances')
        titles.append(title)
        words = [word.get('reg', word.text).lower() for word in word_tags if word.text != None]
        plays.append(words)
    elif title in b_and_f_plays:
        genres.append('Beaumont & Fletcher')
        titles.append(title)
        words = [word.get('reg', word.text).lower() for word in word_tags if word.text != None]
        plays.append(words)

#print(len(titles))
#print(titles)
#print(genres)
#first_play = plays[0]
#most_common_words = collections.Counter(first_play).most_common(25)
#print(most_common_words)

counts = collections.Counter(genres)
#print(counts)
#subset_play = plays[6]
#most_common_words = collections.Counter(subset_play).most_common(100)
#print(most_common_words)
#print(counts)
#fig, ax = plt.subplots()
#ax.bar(counts.keys(), counts.values(), width=0.5)
#ax.set(xlabel='genre', ylabel='count')
#plt.show()

vocabulary = extract_vocabulary(plays, min_count=2)
document_term_matrix = np.array(corpus2tdm(plays, vocabulary))

print(f"document term matrix with "
      f"|D| = {document_term_matrix.shape[0]} documents and "
      f"|V| = {document_term_matrix.shape[1]} words.")

love_idx = vocabulary.index('love')
blood_idx = vocabulary.index('blood')

love_counts = document_term_matrix[:, love_idx]
blood_counts = document_term_matrix[:, blood_idx]

genres = np.array(genres)

fig, ax = plt.subplots(figsize=(20,8))
for genre in ('Comedies', 'Histories', 'Tragedies', 'Late Romances', 'Beaumont & Fletcher'):
    ax.scatter(
        love_counts[genres == genre],
        blood_counts[genres == genre],
        label=genre,
        alpha=0.7
    )
texts = []
for i, txt in enumerate(titles):
    texts.append(ax.annotate(txt, xy=(love_counts[i], blood_counts[i]), xytext=(love_counts[i],blood_counts[i]+.3)))
adjust_text(texts)

#for i, txt in enumerate(titles):
    #ax.annotate(txt, (love_counts[i], blood_counts[i]))
ax.set(xlabel='love', ylabel='blood')
plt.legend()
plt.show()

tragedy_means = document_term_matrix[genres == 'Tragedies'].mean(axis=0)
history_means = document_term_matrix[genres == 'Comedies'].mean(axis=0)
comedy_means = document_term_matrix[genres == 'Histories'].mean(axis=0)
late_romances_means = document_term_matrix[genres == 'Late Romances'].mean(axis=0)
beaumont_and_fletcher_means = document_term_matrix[genres == 'Beaumont & Fletcher'].mean(axis=0)

fig, ax = plt.subplots()
ax.scatter(comedy_means[love_idx], comedy_means[blood_idx], label='Comedies')
ax.scatter(tragedy_means[love_idx], tragedy_means[blood_idx], label='Tragedies')
ax.scatter(history_means[love_idx], history_means[blood_idx], label='Histories')
ax.scatter(late_romances_means[love_idx], late_romances_means[blood_idx], label='Late Romances')
ax.scatter(beaumont_and_fletcher_means[love_idx], beaumont_and_fletcher_means[blood_idx], label='Beaumont & Fletcher')

ax.set(xlabel='love', ylabel='blood')
plt.legend()
plt.show()

tragedy = np.array([tragedy_means[love_idx], tragedy_means[blood_idx]])
comedy = np.array([comedy_means[love_idx], comedy_means[blood_idx]])
late_romances = np.array([late_romances_means[love_idx], late_romances_means[blood_idx]])
history = np.array([history_means[love_idx], history_means[blood_idx]])
beaumont_and_fletcher = np.array([beaumont_and_fletcher_means[love_idx], beaumont_and_fletcher_means[blood_idx]])

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a-b)**2))

def vector_len(v):
    return np.sqrt(np.sum(v**2))

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (vector_len(a) * vector_len(b))

def city_block_distance(a, b):
    return np.abs(a-b).sum()

def nearest_neighbors(X, metric='cosine'):
    distances = dist.pdist(X, metric=metric)
    distances = dist.squareform(distances)
    np.fill_diagonal(distances, np.inf)
    return distances.argmin(1)

genre_vectors = {
    'Comedies': comedy_means,
    'Tragedies': tragedy_means,
    'Late Romances': late_romances_means,
    'Histories': history_means,
    'Beaumont & Fletcher': beaumont_and_fletcher_means
}

metrics = {
    'cosine': dist.cosine,
    'manhattan': dist.cityblock,
    'euclidean': dist.euclidean
}

for metric_name, metric_fn in metrics.items():
    print(metric_name)
    for v1, v2 in itertools.combinations(genre_vectors, 2):
        distance = metric_fn(genre_vectors[v1], genre_vectors[v2])
        print(f'       {v1} - {v2}: {distance:.2f}')

neighbor_indices = nearest_neighbors(document_term_matrix)
nn_genres = genres[neighbor_indices]
print(nn_genres[:5])

overlap = np.sum(genres == nn_genres)
print(f'Matching pairs (normalized): {overlap / len(genres):.2f}')

print(collections.Counter(nn_genres[genres == 'Tragedies']).most_common())
print(collections.Counter(nn_genres[genres == 'Comedies']).most_common())
print(collections.Counter(nn_genres[genres == 'Late Romances']).most_common())
print(collections.Counter(nn_genres[genres == 'Histories']).most_common())
print(collections.Counter(nn_genres[genres == 'Beaumont & Fletcher']).most_common())

t_dists, c_dists, h_dists, b_and_f_dists = [], [], [], []
for lr in document_term_matrix[genres == 'Late Romances']:
    t_dists.append(cosine_distance(lr, tragedy_means))
    c_dists.append(cosine_distance(lr, comedy_means))
    h_dists.append(cosine_distance(lr, history_means))
    b_and_f_dists.append(cosine_distance(lr, beaumont_and_fletcher_means))

print(f'Mean distance to Comedy vector: {np.mean(c_dists):.3f}')
print(f'Mean distance to Tragedy vector: {np.mean(t_dists):.3f}')
print(f'Mean distance to History vector: {np.mean(h_dists):.3f}')
print(f'Mean distance to Beaumont & Fletcher vector: {np.mean(b_and_f_dists):.3f}')

fig, ax = plt.subplots()
ax.boxplot([t_dists, c_dists, h_dists, b_and_f_dists])
ax.set(xticklabels=('Tragedies', 'Comedies', 'Histories', 'Beaumont & Fletcher'), ylabel='Distances to genre mean')
plt.show()

c_dists = np.array(c_dists)
outliers = c_dists.argsort()[::-1][:2]

late_romances_titles = np.array(titles)[genres == 'Late Romances']
print('\n'.join(late_romances_titles[outliers]))
