import Image
import random
import math
import nltk
import sys


new_image = Image.new("RGB", (100, 100))

x = 0
y = 0

for x in range(0, 100):
    new_image.putpixel((x,0), (255,0,0))
#    for y in range(0, sq_size):
#        word = book_words[i]
#    
#        if i < (len_bw - 1):
#            i += 1     
#        else:
#            break
#
#
#        if not words_colors.has_key(word):
#            random_color = (random.randrange(0,255),
#                            random.randrange(0,255),
#                            random.randrange(0,255))
#            words_colors[word] = random_color
#        

new_image.save("t.png")	    


