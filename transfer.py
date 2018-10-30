#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 2018/10/17

import os

f = open('dict.txt', mode = 'w+', encoding = 'utf-8')
fs = os.walk('/Users/shane/Downloads/2018/')
laji = [fn for fn in fs]
laji = laji[0]
print(laji)
root, files = laji[0], laji[-1]
for file in files:
    fn = os.path.join(root, file)
    print(fn)
    with open(fn, mode = 'r', encoding = 'utf-8') as ff:
        for line in ff:
            f.write(line.strip() +'\n')
