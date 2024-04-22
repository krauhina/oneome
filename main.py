from PIL import Image
import os

im = Image.open("FOTO.jpg")
#(width, height) = im.size

def cropping():
    height_percent = (28 / float(im.size[1]))
    width = int((float(im.size[0]) * float(height_percent)))
    new_im = im.resize((width, 28))

    x = 1
    im_len = (width//19) + 1
    for i in range(1, im_len):
        im_crop = new_im.crop((x, 1, 20*i, 28))
        x += 20
        im_crop.show()
        chebe(im_crop)

def chebe(img):
    img.thumbnail((28, 28))
    blackAndWhite = img.convert("L")
    blackAndWhite.save("bw.jpg")
    bw = Image.open("bw.jpg")
    itog = []
    x_g = []
    for y in range(28):
        for x in range(19):
            color = bw.getpixel((x, y))
            if color <= 30:
                ro = '1'
            else:
                ro = '0'
            x_g.append(ro)
            if x == 18:
                itog.append(x_g)
                x_g = []
    os.remove("bw.jpg")
    return print(itog)

cropping()
