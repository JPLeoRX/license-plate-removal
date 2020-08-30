from PIL import Image as imageMain
from PIL.Image import Image

imagePath = '../sample-images/1.jpg'
imagePil = imageMain.open(imagePath)
imagePil.show()