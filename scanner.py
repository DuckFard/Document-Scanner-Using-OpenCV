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
#Setting up contour
def order_points(pts): #getting coordinates of the document's rectangle's 4 corners
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts,axis=1)
    rect[1] = pts[np.argmin(diff)]
    return rect
def four_point_trans(image, pts):
    rect = order_points(pts)
    (tl,tr,br,bl) = rect
    width1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width2 = np.sqrt(((tr[0]-tl[0])**2) + ((br[1]-bl[1])**2))
    maxwidth = max(int(width1),int(width2))
    height1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxheight = max(int(height1), int(height2))
    dst = np.array([
        [0,0],
        [maxwidth - 1,0] 
        [maxwidth - 1, maxheight-1]
        [0, maxheight - 1]],
        dtype="float32") 
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxwidth, maxheight))
    return warped
