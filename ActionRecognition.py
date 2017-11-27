import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as si
from PIL import Image
import deriv, math
import numpy.linalg as lin
import cv2

def hog(img):
    img = cv2.resize(img, (64, 128), interpolation = cv2.INTER_AREA)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0)
	Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
	mag, ang = cv2.cartToPolar(Ix, Iy)
