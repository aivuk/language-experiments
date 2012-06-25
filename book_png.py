import Image
import random
import math
import nltk
import sys

def random_colored_words(words, image_filename):
    # Create the image
    sq_size = int(math.ceil(math.sqrt(len(words))))
    new_image = Image.new("RGB", (sq_size, sq_size))

    x = 0
    y = 0
    
    i = 0 
    words_colors = {}
    len_bw = len(words)

    for y in range(0,sq_size):
        for x in range(0, sq_size):
            word = words[i]
        
            if i < (len_bw - 1):
                i += 1     
            else:
                break
    
    
            if not words_colors.has_key(word):
                random_color = (random.randrange(0,255),
                                random.randrange(0,255),
                                random.randrange(0,255))
                words_colors[word] = random_color
            
            new_image.putpixel((x,y), words_colors[word])

    new_image.save(image_filename)	    

def blue_red_gradient_words(words, image_filename):
    # Create the image
    sq_size = int(math.ceil(math.sqrt(len(words))))

    x = 0
    y = 0
    
    i = 0 
    words_colors = {}

    words_freq = nltk.FreqDist(words)

    nword_max = math.log(words_freq[words_freq.max()])

    new_image = Image.new("RGB", (sq_size, sq_size))

    for y in range(0,sq_size):
        for x in range(0, sq_size):
            word = words[i]
            if i < len(words) - 1:
                i += 1     
            else:
                break
    
    
      	#    if words_freq[word] > 100:
	#         color = (0,0,0)
    	#    else:
            color = (int(math.log(words_freq[word])/nword_max*255), 0, int(255 - math.log(words_freq[word])/nword_max*255))
            
	    def draw_word(new_image, word, color, p):
            	new_image.putpixel((x,y), color)

	    draw_word(new_image, word, color, 0)

    new_image.save(image_filename)	    


book_filename = sys.argv[1]

book_fp = open(book_filename)
book = book_fp.read()

book_words = nltk.word_tokenize(book)

blue_red_gradient_words(book_words, book_filename[:-3] + "png")
random_colored_words(book_words, book_filename[:-3] + "-random.png")



