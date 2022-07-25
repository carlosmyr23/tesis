# -*- coding: utf-8 -*-
import os
import sys
from typing import TextIO
#os.system('pip install nlp_rake')
#os.system('pip install wordcloud')
#sys.path.append('/home/carlos/.local/lib/python3.9/site-packages')
from nlp_rake import Rake

#  abre texto limpio aún en texto.txt
with open('texto.txt', 'r') as f2:
    data = f2.read()
    print(data)
texto = data

import nlp_rake

rake = Rake(
    min_chars=3,
    max_words=3,
    min_freq=3,
    language_code= 'es',  # 'en'
    stopwords= None, # {' que ',' si ','que', ' la ', ' una ', ' de ', ' y ', ' en ', 'y', ' más ', ' los ',' es ' },     # {'and', 'of'}
    lang_detect_threshold=90,
    max_words_unknown_lang=2,
    generated_stopwords_percentile=80,
    generated_stopwords_max_len=3,
    generated_stopwords_min_freq=2,
)

keywords = rake.apply(
    texto,
    text_for_stopwords=None,
)

res = keywords
res
print(res)

#os.system('mkdir Carpeta_test')
#os.system('cd Carpeta_test')
#os.system('touch archivo.py')
from wordcloud import WordCloud
import matplotlib.pyplot as plt
wc = WordCloud(background_color='white',width=3200,height=2400)
plt.figure(figsize=(15,7))
plt.imshow(wc.generate_from_frequencies({ k:v for k,v in res }))
plt.figure(figsize=(15,7))
plt.imshow(wc.generate(texto))
wc.generate(texto).to_file('ds_wordcloud_test.png')


