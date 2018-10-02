# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 06:26:28 2018

@author: ximing
"""

QQQprevClose=175.82
QQQOpen = float(input("What is QQQOpen? "))
print(type(QQQOpen))
TQQQprevClose = 61.5
print(TQQQprevClose*((QQQOpen*1.0004/QQQprevClose-1)*3+1))