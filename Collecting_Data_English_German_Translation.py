# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 13:54:14 2018

@author: J.Wan
"""

# This Program is Used for Collecting the English-German Sentence Translation Data
import requests
import io
from zipfile import ZipFile
The_sentence_url = 'http://www.manythings.org/anki/deu-eng.zip'
r = requests.get(The_sentence_url)
z = ZipFile(io.BytesIO(r.content))
file = z.read('deu.txt')
# Now we need to format Data
eng_ger_data = file.decode()
eng_ger_data = eng_ger_data.encode('ascii',errors='ignore')
eng_ger_data = eng_ger_data.decode().split('\n')
eng_ger_data = [x.split('\t') for x in eng_ger_data if len(x)>=1]
[english_sentence, german_sentence] = [list(x) for x in zip(*eng_ger_data)]
print(len(english_sentence))
print(len(german_sentence))
print(eng_ger_data[10])
print("------------")
print(eng_ger_data[20])
print("------------")
print(eng_ger_data[30])
