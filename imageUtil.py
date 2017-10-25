import os, sys
from PIL import Image, ImageOps

size = 128, 128
rootdir = "./flower_photos"
resizedir = "./resized_photos"
for subdir, dirs, files in os.walk(rootdir):
    for dir in dirs:
        for subdir1, dirs1, files1 in os.walk(rootdir + "/" + dir):
            for file in files1:
                try:
                    im = Image.open(rootdir + "/" + dir + "/"+ file)
                    im.thumbnail(size, Image.ANTIALIAS)
                    thumb = ImageOps.fit(im, size, Image.ANTIALIAS)
                    thumb.save(resizedir + "/" + dir + "/"+ file,"JPEG")
                except IOError:
                    print "cannot create thumbnail for '%s'aaa", IOError.message #% infile
    # for file in files:
    #     print os.path.join(subdir, file)
# for infile in sys.argv[1:]:
#     outfile = os.path.splitext(infile)[0] + ".thumbnail"
#     if infile != outfile:
#         try:
#             im = Image.open(infile)
#             im.thumbnail(size, Image.ANTIALIAS)
#             im.save(outfile, "JPEG")
#         except IOError:
#             print "cannot create thumbnail for '%s'" % infile
# for infile in sys.argv[1:]:
fileName = "1879567877_8ed2a5faa7_n.jpg"
# if infile != outfile:
try:
    im = Image.open("/1/dl2017proj2/flower_photos/daisy/"+fileName)
    im.thumbnail(size, Image.ANTIALIAS)
    thumb = ImageOps.fit(im, size, Image.ANTIALIAS)
    thumb.save("resized_photos/" + fileName, "JPEG")
except IOError:
    print "cannot create thumbnail for '%s'aaa", IOError.message #% infile