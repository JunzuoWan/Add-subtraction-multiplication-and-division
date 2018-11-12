# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 13:56:21 2018
The program is mainly based on https://github.com/nfmcclure/tensorflow_cookbook
"""

# The Works of Shakespeare Data
import requests

shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
# Get Shakespeare text
response = requests.get(shakespeare_url)
shakespeare_file = response.content
# Decode binary into string
shakespeare_text = shakespeare_file.decode('utf-8')
# Drop first few descriptive paragraphs.
shakespeare_text = shakespeare_text[7675:]
print(len(shakespeare_text))
