import cv2
import numpy as np
import argparse
#setting up argparser 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
args = vars(ap.parse_args())
#image input
image = cv2.imread(args["image"])
orig = image.copy()
#Preprocessing
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY) #Step1: Grayscale
gray = cv2.GaussianBlur(gray,(5,5),0) #Step2: reduce noise
edged = cv2.Canny(gray,75,200)


