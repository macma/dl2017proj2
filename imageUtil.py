import os
import sys
from PIL import Image, ImageOps

size = 128, 128
rootdir = "./flower_photos"
resizedir = "./resized_photos"
for subdir, dirs, files in os.walk(rootdir):
    for dir in dirs:
        for subdir1, dirs1, files1 in os.walk(rootdir + "/" + dir):
            for file in files1:
                try:
                    im = Image.open(rootdir + "/" + dir + "/" + file)
                    im.thumbnail(size, Image.ANTIALIAS)
                    thumb = ImageOps.fit(im, size, Image.ANTIALIAS)
                    thumb.save(resizedir + "/" + dir + "/" + file, "JPEG")
                except IOError:
                    print "cannot create thumbnail for '%s'aaa", IOError.message  # %aa infile
