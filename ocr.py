import pytesseract
import argparse
import cv2
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
import nltk
import re 
from nltk.tokenize import word_tokenize 
from nltk.corpus import wordnet
from PIL import Image


def tesseract_ocr(image_dir):
    pytesseract.pytesseract.tesseract_cmd = "Tesseract-OCR/tesseract.exe"
    image = cv2.imread(image_dir,0 )
    return pytesseract.image_to_string(image).lower()
def fitur(ocr_text):
    nltk.download('punkt',quiet=True)
    sent_tokens = nltk.sent_tokenize(ocr_text)
    nyari_pakain=[]
    for i in sent_tokens:
        if "kasir" in i:
                nyari_pakain.append(i)
    nyari_pakain=nyari_pakain[0].split("\n")
    hasil=[]
    for i in range(len(nyari_pakain)):
        hasil.append(nyari_pakain[i-1])
        if "lotal" in nyari_pakain[i]:
            break
    kasir=[]
    for i in range(len(hasil)):
        kasir.append(hasil[i])
        if "kasir" in hasil[i]:
                break
    tokens = [token for token in hasil if token not in kasir]
    del tokens[0], tokens[-1]
    return tokens

def barang(fitur):
    words = []            
    for i in range(len(fitur)):
        j = word_tokenize(fitur[i])+["angka"]
        for k in range(len(j)):
            if j[k].isalnum():
                words.append(j[k])
    f = " ".join(words)
    f = f.split("angka")
    del f[-1]
    return f
        
def harga(fitur):
    selain_words = []
    for i in range(len(fitur)):
        j = word_tokenize(fitur[i])+["999,999"]
        for k in range(len(j)):
            if re.findall(r'([0-9]{1,3}(?=\,))',j[k]):
                selain_words.append(j[k])
    harganya=[]
    for i in range(len(selain_words)):
        if selain_words[i]=="999,999":
            harganya.append(selain_words[i-1])
            continue
    return harganya