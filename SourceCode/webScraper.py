from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import re
import time
import urllib.request
import json
import pandas as pd
import os
import random

accounts = [
    #"stevint",
    ###"danielkordan",
    #"alliemtaylor",
    #"alexstrohl",
    #"hannes_becker",
    ###"fursty",
    #"jasoncharleshill",
    # "shortstache",
    #"rodtrvn",
    #"jannikobenhoff",
    #"rawmeyn",
    ###"paulnicklen",
    ###"airpixels",
    #"dudelum",
    #"lyesk",
    #"aaronbhall",
    
    #"emmett_sparling",
    #"taylormichaelburk",
    #"thiswildidea",
    #"donjay",
    #"jonpauldouglass",
    #"brianladder",
    #"kelianne",
    #"haakeaulana",
    #"elliepritts",
    #"edkashi",
    #"adrienneraquel",
    #"CristinaMittermeier",
    #"danielmudliar",
    #"vaderbreath",
    #"davidalanharvey",
    
    #"mmuheisen",
    #"pangea",
    
    #"beverlyjoubert",
    ###"loki",
    #"jordanherschel",
    #"lebackpacker",
    #"jackrmoriarty",
    #"nathanaelbillings",
    #"willbl",
    #"tiffpenguin"
    #"masonstrehl",
    #"andrewtkearns",
    ###"jessfindlay",
    #"mattysmithphoto",
    #"davidlloyd",
    #"cschoonover",
    ###"cestmaria",
    #"sophlog",
    #"macenzo",
    ###"helloemilie",
    #"laurenepbath",
    #"theblondeabroad",
    #"maria.svarbova",
    #"lavicvic",
    #"hirozzzz",
    #"garethpon",
    #"kevinruss",
    #"andrewknapp",
    #"samhorine",
    #"kirstenalana",
    #"reallykindofamazing",
    #"palchenkov",
    #"heysp",
    #"danrubin",
    #"joelsartore",
    #"amivitale",
    #"benlowy",
    #"stefanounterthiner",
    #"yamashitaphoto",
    #"renan_ozturk",
    #"evolumina",
]

for account in accounts:
    url = "https://www.instagram.com/{}/".format(account)
    img_set = set()

    # create a new Chrome session
    driver = webdriver.Chrome()
    driver.implicitly_wait(30)
    driver.get(url)
    time.sleep(3)

    scrollCount = 0
    lastHeight = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        newHeight = driver.execute_script("return document.body.scrollHeight")
        if newHeight == lastHeight:
            break
        lastHeight = newHeight
        scrollCount += 2

    driver.execute_script("window.scrollTo(0, 0);")
    for x in range(0, lastHeight, 750):
        driver.execute_script("window.scrollTo(0, {});".format(x))
        time.sleep(1)

        tempImgs = driver.find_elements_by_class_name("KL4Bh")
        for i in tempImgs:
            img_src = i.find_element_by_tag_name('img').get_attribute('src')
            img_set.add(img_src)

        if len(img_set) > 750:
            break

    print(len(img_set))
    driver.close()

    img_list = list(img_set)
    random.shuffle(img_list)
    img_count = 0
    for img in img_list:
        if img_count < 76:
            urllib.request.urlretrieve(img, "10users\data\\validation\{}\{}-{}.jpg".format(account, account, img_count))
        else:
            urllib.request.urlretrieve(img, "10users\data\\train\{}\{}-{}.jpg".format(account, account, img_count))
        img_count += 1
