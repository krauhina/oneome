from PIL import Image
import os

im = Image.open("FOTO.jpg")
(width, height) = im.size
height = 135

height_percent = (height / float(im.size[1]))
width = int((float(im.size[0]) * float(height_percent)))
new_im = im.resize((width, height))

def cropping():
    k = 1
    x = 20
    max = width//100 + 1
    for i in range(10000):
        im_crop = new_im.crop((x, 10, 100*k, 130)) #переменная содержит обрезанную фотографию буквы
        x += 100
        k += 1
        chebe(im_crop)

        if k == max:
            return


def chebe(im_crop):
    im_crop.thumbnail((28, 28))
    blackAndWhite = im_crop.convert("L")
    blackAndWhite.save("bw.jpg")
    bw = Image.open("bw.jpg")
    itog = []
    x_g = []
    for y in range(28):
        for x in range(18):
            color = bw.getpixel((x, y))
            if color <= 30:
                ro = '1'
            else:
                ro = '0'
            x_g.append(ro)
            if x == 17:
                itog.append(x_g)
                x_g = []
    os.remove("bw.jpg")
    return itog

cropping()