from lxml import etree
import pandas as pd
from collections import Counter
nsmap = {'tei': 'http://www.tei-c.org/ns/1.0'}

parser = etree.XMLParser(collect_ids=False)
tree = etree.parse('./datasets/A27177_07.xml', parser)
beggars_bush = tree.getroot()

all_word_tags = beggars_bush.findall(".//tei:w", namespaces=nsmap)
all_words = [w.text for w in all_word_tags]
#print(all_words[:25])

all_regularized = [w.get('reg', w.text) for w in all_word_tags]
#print(all_regularized[:25])

he_tag = all_word_tags[24]
#print(etree.tostring(he_tag))

#print(he_tag.text, he_tag.attrib['lemma'], he_tag.attrib['pos'])

all_word_info = [(w.text, w.attrib.get('reg', w.text), w.attrib.get('lemma'), w.attrib.get('pos')) for w in all_word_tags[:10]]
#print(all_word_info)

all_nouns = [w.text for w in all_word_tags if w.get('pos').startswith('n')]
#print(Counter(all_nouns).most_common()[:10])

all_line_tags = beggars_bush.findall(".//tei:l", namespaces=nsmap)
words_by_line = [[w.text for w in l.findall(".//tei:w", namespaces=nsmap)] for l in all_line_tags]
#print(words_by_line[:20])

#for line in all_line_tags[:20]:
    #print(' '.join([w.text for w in line.findall(".//tei:w", namespaces=nsmap)]))

#for line in all_line_tags[:20]:
    #print(' '.join([child.text for child in line]))

w_and_pc = beggars_bush.xpath("//tei:w|//tei:pc", namespaces=nsmap)
all_sentences = []
new_sentence = []

for tag in w_and_pc[:500]:
    if 'unit' in tag.attrib and tag.get('unit') == 'sentence':
        if tag.text != None:
            new_sentence.append(tag.text)
        all_sentences.append(new_sentence)
        new_sentence = []
    else:
        new_sentence.append(tag.text)

#print(all_sentences)

all_divs = beggars_bush.xpath("//tei:div[@type='act']|//tei:div[@type='scene']", namespaces=nsmap)
for div in all_divs:
    print(div.attrib)

#title_page = beggars_bush.find(".//tei:div[@type='play']", namespaces=nsmap)
#words_on_title_page = [w.text for w in title_page.findall(".//tei:w", namespaces=nsmap)]
#print(words_on_title_page)

# Lets find things in Act 3:
act3 = beggars_bush.find(".//tei:div[@type='act'][@n='3']", namespaces=nsmap)
words_in_act_3 = [w.text for w in act3.findall(".//tei:w", namespaces=nsmap)]
#print(words_in_act_3)

all_acts = beggars_bush.findall(".//tei:div[@type='act']", namespaces=nsmap)
words_by_division = [[w.text for w in part.findall(".//tei:w", namespaces=nsmap)] for part in all_divs]

# Let's print the first ten words by play division
for act in words_by_division:
    print(act[:10])

# Let's print out a .csv file with all the nouns in each division of the play:

nouns_by_division = [[w.get('reg', w.text) for w in part.findall(".//tei:w", namespaces=nsmap) if w.get("pos").startswith("n")] for part in all_divs]
nouns_counts_by_division = [Counter(noun_list) for noun_list in nouns_by_division]

divisions = beggars_bush.find(".//tei:div[@type='act']", namespaces=nsmap)
number_of_divisions = len(all_divs)
#print(number_of_divisions)

division_names = [f"Division {number}" for number in range(number_of_divisions)]
noun_counts_df = pd.DataFrame(nouns_counts_by_division, index=division_names).fillna(0)
noun_counts_df = noun_counts_df.T
print(noun_counts_df.head(50))
