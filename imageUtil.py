import os
import sys
from PIL import Image, ImageOps
import shutil
import numpy as np
import cPickle as pickle

imgpixel = 28
size = imgpixel, imgpixel
rootdir = "./flower_photos"
resizedir = "./resized_photos" + str(imgpixel)

def removeFolder():
    if os.path.exists(resizedir):
        shutil.rmtree(resizedir)
    createDir(resizedir);
    createDir(resizedir + "/tulips");
    createDir(resizedir + "/test");
    createDir(resizedir + "/sunflowers");
    createDir(resizedir + "/roses");
    createDir(resizedir + "/dandelion");
    createDir(resizedir + "/daisy");
    
def createDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
def resizeSinglePhoto(input, output):
        im = Image.open(input)
        im.thumbnail(size, Image.ANTIALIAS)
        thumb = ImageOps.fit(im, size, Image.ANTIALIAS)
        thumb.save(output, "JPEG")
def resizePhoto():
    for subdir, dirs, files in os.walk(rootdir):
        datacounter = 0
        for dir in dirs:
            for subdir1, dirs1, files1 in os.walk(rootdir + "/" + dir):
                for file in files1:
                    im = Image.open(rootdir + "/" + dir + "/" + file)
                    im.thumbnail(size, Image.ANTIALIAS)
                    thumb = ImageOps.fit(im, size, Image.ANTIALIAS)
                    print resizedir + "/" + dir + "/" + file
                    thumb.save(resizedir + "/" + dir + "/" + file, "JPEG")
if __name__ == "__main__":
    removeFolder();
    resizePhoto();
