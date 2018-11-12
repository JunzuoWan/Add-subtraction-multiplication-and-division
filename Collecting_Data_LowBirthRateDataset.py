# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:18:42 2018
The program is mainly based on https://github.com/nfmcclure/tensorflow_cookbook
"""

from tensorflow.python.framework import ops
ops.reset_default_graph()

# Low Birthrate Data
import requests

birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
birth_file = requests.get(birthdata_url)
birth_data = birth_file.text.split('\r\n')
birth_header = birth_data[0].split('\t')
birth_data = [[float(x) for x in y.split('\t') if len(x)>=1] for y in birth_data[1:] if len(y)>=1]
print(len(birth_data))
print(len(birth_data[0]))

