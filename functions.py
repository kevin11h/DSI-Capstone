# funtions for image pre-processing  *** should create one for resizing 
from operator import itemgetter
from collections import Counter

from PIL import Image, ImageFilter
from IPython.display import Image as imgp
import cv2, pytesseract
from matplotlib.pyplot import imshow

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

import pandas as pd, numpy as np
import seaborn as sns, matplotlib.pyplot as plt

import requests
from selenium import webdriver
from bs4 import BeautifulSoup

import re

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from statistics import mode

import math
# import smtplib
# import time

#%matplotlib inline


def image_rotate(image_r, angle):
    
    w = image_r.shape[1]
    h = image_r.shape[0]

    if angle not in [90,180,270]: return(image_r)
    
    if(angle == 90):
        center = (w // 2, w // 2)
        M = cv2.getRotationMatrix2D(center, 90, 1.0)
        rotated = cv2.warpAffine(image_r, M, (h, w), flags=cv2.INTER_CUBIC)
        
    elif(angle == 180):
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        rotated = cv2.warpAffine(image_r, M, (w, h), flags=cv2.INTER_CUBIC)
        
    elif(angle==270):        
        center = (h // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 270, 1.0)
        rotated = cv2.warpAffine(image_r, M, (h, w), flags=cv2.INTER_CUBIC)
      
    return(rotated)


## -- for analyzing an image / book


def parse_book(book_image, equ = False, rotate = False, new_width = 900):
    
    # resize
    book_image = cv2.resize(book_image,  (new_width, int(book_image.shape[0]*new_width/book_image.shape[1]) )) 

    # rotate
    if rotate: 
        book_image = image_rotate(book_image, 180)
#         preview(book_image)

    # equalize 
    if equ: book_image = cv2.equalizeHist(book_image)
        
    parse = book_image

    config = '--eom 12 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzQWERTYUIOPASDFGHJKLZXCVBNM'

    parsed = pytesseract.image_to_string(parse, config = config)
    parsed_raw = parsed.split('\n')

    #     #-- char height
    #     # boxes = pytesseract.image_to_boxes(parse, output_type='dict', nice = 1, config = config)
    #     # try: charh = mode(boxes['top']) - mode(boxes['bottom'])
    #     # except: charh = 0
    #     # print(charh)
    
    return(parsed_raw)


def lines_detect(img): 
    
    minLineLength = 50
    maxLineGap = 200
    
    blur= cv2.GaussianBlur(img, (5, 5), 10)
    edged = cv2.Canny(blur, 100, 200)
    kernel = np.ones((8,8),np.uint8)
    dilation = cv2.dilate(edged,kernel,iterations = 1)

    lines = cv2.HoughLinesP(dilation,1,np.pi/180,300,minLineLength,maxLineGap)

    return(lines)



def lines_detect_2(img): ## *** need to test these two 
    
    minLineLength = 5000
    maxLineGap = 100
    
    blur= cv2.GaussianBlur(img, (5, 5), 100)
    edged = cv2.Canny(blur, 100, 200)
    kernel = np.ones((8,8),np.uint8)
    dilation = cv2.dilate(edged,kernel,iterations = 1)

    lines = cv2.HoughLinesP(dilation,1,np.pi/180,400,minLineLength,maxLineGap)

    return(lines)


def parsed_analyze(parsed):
    
    ### see how many books we're expecting from this image
    ### (sometimes the cut does not go as smooth and two or more books end up on an image,
    ### in which case, pytesseract will most likely add an empty entry in the list. This could also happend
    ### if book's title has two lines.. )
    ### note: currently i am going to just have 1 & 2 books option.. i can work on bettering this part later

    expect = 1
    space = [i for i,x in enumerate(parsed) if not x]
    if space: expect = 2

    ## remove lines that don't have at least 4 letter blocks 
    pattern = re.compile("[a-zA-Z]{4,}")
    lines = [x for x in parsed if(pattern.search(x))]
    
    return(lines, expect, not lines)


# main one -- parsing text from an image
def image_parse(book_image, equ = False, rotate = False, new_width = 900):
    
    # resize
    book_image = cv2.resize(book_image,  (new_width, int(book_image.shape[0]*new_width/book_image.shape[1]) )) 

    # rotate
    if rotate: 
        book_image = image_rotate(book_image, 180)
#         preview(book_image)

    # equalize 
    if equ: book_image = cv2.equalizeHist(book_image)
        
    parse = book_image

    config = '--eom 12 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzQWERTYUIOPASDFGHJKLZXCVBNM'

    parsed = pytesseract.image_to_string(parse, config = config)
    parsed_raw = parsed.split('\n')
    
    return(parsed_raw)



## -- for plotting and previewing images

def preview(img, gray = False):
    
    cmap = None
    if gray == True: cmap = plt.cm.gray   
    plt.figure(figsize = (10,10)) 
    plt.imshow(img, cmap = cmap)

def plot_hline(img, y, color = (0,255,0)):
    
    cv2.line(img,(0,y),(img.shape[1],y),color,7)
    return(img)
    
def plot_lines(img, lines, color = (0,255,0)):
    
    lines = lines.reshape(lines.shape[0],4)
    
    for x in range(0, len(lines)):
        
        x1,y1,x2,y2 = lines[x]
#         print("{} {} {} {}".format(x1,y1,x2,y2))
        cv2.line(img,(x1,y1),(x2,y2),color,7)
    
    return(img)


def place_dot(image, x,y, color = (0,255,0)):
    
    h = image.shape[0]
    w = image.shape[1]
    
    image = cv2.circle(image,(int(x),h-int(y)),4,color,-11)
    return(image)


def detect_angle(line):   
    x1, y1, x2, y2 = line
    return(np.arctan((y2 - y1)/(x2 - x1))*180/np.pi)


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext




    
    
def amazon_parse(source, output = False):
       
        global item_id
        item_id = len(items)
        items.append({})

#         items[item_id]['url'] = s
        if output: print("\t \t ---->> url")

        soup = BeautifulSoup(source, "lxml")  
        
        ## -- determine type 
        
        if(soup.find('meta', {'content': 'on'})): amazon_type = 1
        else: amazon_type = 2
       
        items[item_id]['type'] = amazon_type
        
        ## --- DESCRIPTION
            
        descr_raw = soup.find('noscript').find_next('noscript').text.strip()
        descr = cleanhtml(descr_raw).strip('\n').strip('\t').strip('\\').strip('&amp;')
        
        if(descr==''):
            descr_raw = soup.find('noscript').find_next('noscript').find_next('noscript').text.strip()
            descr = cleanhtml(descr_raw).strip('\n').strip('\t').strip('\\').strip('&amp;')

        items[item_id]['desription'] = descr 
        if output: print("\t \t ---->> description")
        
    
        try:
     
            try:
                image_url = soup.find("img", {"class": "frontImage"}).attrs['src']
        
            except:
                
                image = soup.find("div", {"id": "mainImageContainer"})
                image_url = image.find('img').attrs['src']
                

            if '.jpg' in image_url: 

                if output: print(" *********** IMAGE ************** ")
                
                items[item_id]['img_url'] = image_url
                # save_path = 'images/'+item_id+'.jpg'
                save_path = 'images/'+str(item_id)+'.jpg'
                
                img_data = requests.get(image_url).content
                with open(save_path, 'wb') as handler:

                    handler.write(img_data)
                    items[item_id]['image'] = 'saved'
                    
            else: items[item_id]['image'] = 'none'
                
            if output: print("\t \t ---->> image")   
            
        except: pass

        ## --- TITLE
        
        title_tag = soup.find("span", {"id": "productTitle"})
        title = title_tag.text

        items[item_id]['title'] = title
        if output: print("\t \t ---->> title")
        
        ## --- AUTHOR
        
        span = soup.find("span", {"class": "author"})
        author = span.find('a').text

        items[item_id]['author'] = author 
        if output: print("\t \t ---->> author")
        
        return(True)





## -- functions for a new version of rotation detection 

def hor_lines(image): ## detect in which image rotation we have more horizontal lines then vertical
    
    new_width = 700 

    this_image  = cv2.resize(image,  (int(image.shape[1]* new_width/image.shape[0]), new_width )) 

    im = []
    im.append(this_image)
    im.append(image_rotate(this_image, 90))

    res = []

    for i, img in enumerate(im):

        lines = lines_detect(img)
        lines = lines.reshape(lines.shape[0],4)

        co = Counter([(line[2]-line[0]) > 100 for line in lines])
        res.append(co[1]/(co[0]+co[1]))
        
    return(res)



import pickle
from imports import gib_detect_train

model_data = pickle.load(open('imports/gib_model.pki', 'rb'))

def detect(l):
    model_mat = model_data['mat']
    threshold = model_data['thresh']
    return(gib_detect_train.avg_transition_prob(l, model_mat) > threshold)


def detect_gibberish(image):  ## detect probability we're NOT getting gibberish on image vs rotated 180

    res = []
    im = []
    im.append(image)
    im.append(image_rotate(image, 180))

    for i, img in enumerate(im):

        parsed = image_parse(img)

        pattern = re.compile("[a-zA-Z]{4,}")
        lines = [x for x in parsed if(pattern.search(x))]
        co = Counter([detect(x) for x in lines])

        if co[0]+co[1] !=0: res.append(co[1]/(co[0]+co[1]))
        else: res.append(0)
            
    return(res)
