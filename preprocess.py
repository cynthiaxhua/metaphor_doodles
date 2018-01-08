# import the required libraries
import numpy as np
import time
import random
import _pickle as cPickle
import os
import cv2

# set numpy output to something sensible
np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)
# import our command line tools

# libraries required for visualisation:
from IPython.display import SVG, display
import svgwrite
# conda install -c omnia svgwrite=1.1.6 if you don't have this lib
# helper function for draw_strokes

# import our command line tools
import cairosvg
import imageio

def get_bounds(data, factor):
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0
    
  abs_x = 0
  abs_y = 0
  for i in range(len(data)):
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    abs_x += x
    abs_y += y
    min_x = min(min_x, abs_x)
    min_y = min(min_y, abs_y)
    max_x = max(max_x, abs_x)
    max_y = max(max_y, abs_y)
    
  return (min_x, max_x, min_y, max_y)

# little function that displays vector images and saves them to .svg
def draw_strokes_mod(data, factor=0.2, svg_filename = 'sample.svg'):
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  lift_pen = 1
  abs_x = 25 - min_x 
  abs_y = 25 - min_y
  p = "M%s,%s " % (abs_x, abs_y)
  command = "m"
  for i in range(len(data)):
    if (lift_pen == 1):
      command = "m"
    elif (command != "l"):
      command = "l"
    else:
      command = ""
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "
  the_color = "black"
  stroke_width = 1
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  open(svg_filename, "w+")
  dwg.saveas(svg_filename)
  return SVG(dwg.tostring())

def scale_image(infile, size=(48, 48), outfile=None):
    # read image
    img = cv2.imread(infile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)

    h, w = img.shape[:2]
    sh, sw = size

    # aspect ratio of image
    aspect = w/h

    # padding
    pad = [0, 0, 0, 0] # (top, left, bottom, right)

    new_h, new_w = sh, sw

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad[0] = np.floor(pad_vert).astype(int)
        pad[2] = np.ceil(pad_vert).astype(int)

    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad[1] = np.floor(pad_horz).astype(int)
        pad[3] = np.ceil(pad_horz).astype(int)

    # scale and pad
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3], borderType=cv2.BORDER_CONSTANT, value=0)

    # increase contrast
    img[img > 0] = 255

    # display or save as npy
    if not outfile:
        cv2.imshow('scaled image', img)
        cv2.waitKey(0)
    else:
        #np.save(outfile, img)
        return img

#If the file is a .npy file, then a single array is returned.
#If the file is a .npz file, then a dictionary-like object is returned, containing {filename: array} key-value pairs, one for each file in the archive.
#npy_file = np.load("cat.npy")
#npz_file = np.load()

initial = 0
final = 35000

name = "dragon"
npz_data = np.load(name+"/"+name+".npz", encoding="latin1")
train_set = npz_data['train']
valid_set = npz_data['valid']
test_set = npz_data['test']
rough_data = []
#single_img = random.choice(train_set)
result = []
result_initial = initial

for i in range(initial,final):
  current_img = train_set[i]
  svg_img = draw_strokes_mod(current_img,svg_filename=name+"/svgs/img"+str(i)+".svg")
  svginfile = name+"/svgs/img"+str(i)+".svg"
  pngoutfile = name+"/pngs/img"+str(i)+".png"
  cairosvg.svg2png(url=svginfile, write_to=pngoutfile)
  os.remove(svginfile)
  #np.save("valid_"+str(initial)+"_"+str(i)+".npy",result)
  pnginfile = name+"/pngs/img"+str(i)+".png"
  npyoutfile = scale_image(infile=pnginfile, outfile=name+'/npys/img'+str(i)+'.npy')
  if i == initial:
    result = [scale_image(infile=pnginfile, outfile=name+"/npys/img"+str(initial)+".npy")]
  else:
    result = np.concatenate((result,[npyoutfile]),axis = 0)
  os.remove(pnginfile)
  if (i % 5000) == 0:
    print(str(i))
    temp_name = name+"/"+name+"_"+str(initial)+"_"+str(i)+".npy"
    np.save(temp_name,result)
    rough_data.append(temp_name)
    result_initial = i + 1
np.save(name+"/"+name+"_train.npy",result)

result = []
result_initial = 0
for i in range(0,2500):
  current_img = valid_set[i]
  svg_img = draw_strokes_mod(current_img,svg_filename=name+"/svgs/vimg"+str(i)+".svg")
  svginfile = name+"/svgs/vimg"+str(i)+".svg"
  pngoutfile = name+"/pngs/vimg"+str(i)+".png"
  cairosvg.svg2png(url=svginfile, write_to=pngoutfile)
  os.remove(svginfile)
  #np.save("valid_"+str(initial)+"_"+str(i)+".npy",result)
  pnginfile = name+"/pngs/vimg"+str(i)+".png"
  npyoutfile = scale_image(infile=pnginfile, outfile=name+'/npys/vimg'+str(i)+'.npy')
  if i == initial:
    result = [scale_image(infile=pnginfile, outfile=name+"/npys/vimg"+str(initial)+".npy")]
  else:
    result = np.concatenate((result,[npyoutfile]),axis = 0)
  os.remove(pnginfile)
  if (i % 5000) == 0:
    print(str(i))
    np.save(name+"/"+name+"_valid_"+str(initial)+"_"+str(i)+".npy",result)
    result_initial = i + 1
np.save(name+"/"+name+"_valid.npy",result)

result = []
result_initial = 0
for i in range(0,2500):
  current_img = test_set[i]
  svg_img = draw_strokes_mod(current_img,svg_filename=name+"/svgs/timg"+str(i)+".svg")
  svginfile = name+"/svgs/timg"+str(i)+".svg"
  pngoutfile = name+"/pngs/timg"+str(i)+".png"
  cairosvg.svg2png(url=svginfile, write_to=pngoutfile)
  os.remove(svginfile)
  #np.save("valid_"+str(initial)+"_"+str(i)+".npy",result)
  pnginfile = name+"/pngs/timg"+str(i)+".png"
  npyoutfile = scale_image(infile=pnginfile, outfile=name+'/npys/timg'+str(i)+'.npy')
  if i == initial:
    result = [scale_image(infile=pnginfile, outfile=name+"/npys/timg"+str(initial)+".npy")]
  else:
    result = np.concatenate((result,[npyoutfile]),axis = 0)
  os.remove(pnginfile)
  if (i % 5000) == 0:
    print(str(i))
    np.save(name+"/"+name+"_test_"+str(initial)+"_"+str(i)+".npy",result)
    result_initial = i + 1
np.save(name+"/"+name+"_test.npy",result)

train_npy = np.load(name+"/"+name+"_train.npy")
valid_npy = np.load(name+"/"+name+"_valid.npy")
test_npy = np.load(name+"/"+name+"_test.npy")

final = [train_npy,valid_npy,test_npy]
np.save(name+".npy",final)
np.savez(name+"half.npz",train=train_set[:35000],valid=valid_set,test=test_set)

for x in rough_data:
  os.remove(x)

os.remove(name+"/"+name+"_train.npy")
os.remove(name+"/"+name+"_valid.npy")
os.remove(name+"/"+name+"_test.npy")


#for i in range(0,2500):
#  current_img = valid_set[i]
#  svg_img = draw_strokes_mod(current_img,svg_filename=name+"/svgs/vimg"+str(i)+".svg")

#for i in range(0,2500):
#  current_img = test_set[i]
#  svg_img = draw_strokes_mod(current_img,svg_filename=name+"/svgs/timg"+str(i)+".svg")


#for i in range(0,2500):
#  current_img = test_set[i]
#  svg_img = draw_strokes_mod(current_img,svg_filename="svgs/timg"+str(i)+".svg")
#to save subset as .npz file
#np.savez("original.npz",z)
  
# draw a random example (see draw_strokes.py)
#draw_strokes(single_img)
#cairosvg.svg2png(url="sample.svg", write_to='image.png')




