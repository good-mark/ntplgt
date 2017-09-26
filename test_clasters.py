#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import string
import sys
import sqlite3
import xml.dom.minidom
import codecs
from nltk.corpus import stopwords
from collections import deque
codecs.register(lambda name: codecs.lookup('utf-8') if name == 'cp65001' else None)


clusters_db = sqlite3.connect('.\\lihonosov_clusters_full.sqlite')
clusters_cur = clusters_db.cursor()

clusters_cur.execute('SELECT * FROM clusters WHERE class_id=1998')
result = clusters_cur.fetchall()
print '1998:'
for r in result:
	print r[0]