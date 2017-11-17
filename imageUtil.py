import os
import sys
from PIL import Image, ImageOps
import shutil
#from array import *
import numpy as np
import cPickle as pickle

size = 32, 32
rootdir = "./flower_photos"
resizedir = "./resized_photos32"

def removeFolder(flowername):
    if os.path.exists("/1/tensorflow_input_image_by_tfrecord/src/" + flowername):
        shutil.rmtree("/1/tensorflow_input_image_by_tfrecord/src/" + flowername)
def removeTargetFolder():
    removeFolder("tulips");
    removeFolder("test");
    removeFolder("sunflowers");
    removeFolder("roses");
    removeFolder("dandelion");
    removeFolder("daisy");
def copyToFolder(flowername):
    shutil.copytree(resizedir+"/"+flowername, "/1/tensorflow_input_image_by_tfrecord/src/"+flowername);
def copyToTargetFolder():
    copyToFolder("tulips");
    copyToFolder("test");
    copyToFolder("sunflowers");
    copyToFolder("roses");
    copyToFolder("dandelion");
    copyToFolder("daisy");
def createDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

createDir(resizedir);
createDir(resizedir + "/tulips");
createDir(resizedir + "/test");
createDir(resizedir + "/sunflowers");
createDir(resizedir + "/roses");
createDir(resizedir + "/dandelion");
createDir(resizedir + "/daisy");

# data = array('B')
data = [];#np.empty((3119*32*32*3 + 3119), dtype='float32')


def dirToClass(flowername):
    if(flowername == 'tulips'):
        return 1;
    if(flowername == 'sunflowers'):
        return 2;
    if(flowername == 'roses'):
        return 3;
    if(flowername == 'dandelion'):
        return 4;
    if(flowername == 'daisy'):
        return 5;
def resizePhoto():
    for subdir, dirs, files in os.walk(rootdir):
        datacounter = 0
        for dir in dirs:
            for subdir1, dirs1, files1 in os.walk(rootdir + "/" + dir):
                for file in files1:
                    try:
                        im = Image.open(rootdir + "/" + dir + "/" + file)
                        im.thumbnail(size, Image.ANTIALIAS)
                        thumb = ImageOps.fit(im, size, Image.ANTIALIAS)
                        print resizedir + "/" + dir + "/" + file
                        thumb.save(resizedir + "/" + dir + "/" + file, "JPEG")

                        if(dir != 'test'):
                            # data[datacounter] = dirToClass(dir);
                            # datacounter = datacounter + 1;
                            data.append(dirToClass(dir))
                            # data.append(0)
                            im1 = Image.open(resizedir + "/" + dir + "/" + file)
                            pix = im1.load()
                            #then write the rows
                            #Extract RGB from pixels and append
                            #note: first we get red channel, then green then blue
                            #note: no delimeters, just append for all images in the set
                            for color in range(0,3):
                                for x in range(0,32):
                                    for y in range(0,32):
                                        #print color, x, y
                                        # data[datacounter] = pix[x,y][color];
                                        # datacounter = datacounter+1;
                                        data.append(pix[x,y][color])
                            data.append(10)
                            #print datacounter
                    except IOError:
                        print "cannot create thumbnail for '%s'aaa", IOError.message  # %aa infile
    output_file = open('cifar10-ready.bin', 'wb')
    # data.tofile(output_file)
    np.array(data).astype(np.float32).tofile(output_file)
    output_file.close()
    # pickle.dump( data, open( "flowers-train", "wb" ) )
    # output_file.flush()

resizePhoto();
removeTargetFolder();
copyToTargetFolder();
