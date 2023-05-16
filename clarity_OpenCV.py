import cv2

#利用拉普拉斯
def getImageVar(imagepath):
    image = cv2.imread(imagepath)
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
    result = (imageVar)
    return result

