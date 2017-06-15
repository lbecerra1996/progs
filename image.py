from PIL import Image
img = Image.open('imgpy.jpg').convert('L')
img.show()
img.save('brick-house-gs','png')
