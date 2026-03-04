import cv2
import numpy as np
import argparse
import imutils

# 1. Setting up argparser 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
args = vars(ap.parse_args())

# 2. Image input & Ratio calculation
image = cv2.imread(args["image"])
# We keep a copy of the original for the final high-res warp
orig = image.copy()
# Calculate the ratio to scale points back later
ratio = image.shape[0] / 500.0
image = imutils.resize(image, height=500)

# 3. Enhanced Preprocessing for "Busy" Environments
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.bilateralFilter(gray, 9, 75, 75) 
edged = cv2.Canny(blurred, 75, 200)

# IMPORTANT: Dilate the edges to close small gaps in the screen border
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.dilate(edged, kernel, iterations=1)
# --- STEP 4: Refined Contour Finding ---
cnts = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10] # Check more contours

screen_cnt = None

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    # Check if it has 4 points AND is a reasonable size
    if len(approx) == 4:
        # Optional: Check Aspect Ratio to ensure it's a screen/paper shape
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        if 0.5 <= ar <= 2.0: # Adjust based on screen vs paper
            screen_cnt = approx
            break

# Fallback: same as before but ensure np.intp
if screen_cnt is None:
    print("Fallback triggered: Finding largest possible rectangle.")
    rect = cv2.minAreaRect(cnts[0])
    box = cv2.boxPoints(rect)
    screen_cnt = np.intp(box).reshape(4, 1, 2)
    # Final safety: If it's still messy, force a rectangle around it
    if len(screen_cnt) != 4:
        rect = cv2.minAreaRect(cnts[0])
        box = cv2.boxPoints(rect)
        screen_cnt = np.intp(box).reshape(4, 1, 2) 

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # Top-Left
    rect[2] = pts[np.argmax(s)]    # Bottom-Right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-Right
    rect[3] = pts[np.argmax(diff)] # Bottom-Left
    return rect

def four_point_trans(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Calculate Max Width
    width1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxwidth = max(int(width1), int(width2))
    
    # Calculate Max Height
    height1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxheight = max(int(height1), int(height2))
    
    dst = np.array([
        [0, 0],
        [maxwidth - 1, 0],
        [maxwidth - 1, maxheight - 1],
        [0, maxheight - 1]], dtype="float32") 
        
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxwidth, maxheight))

# --- STEP 6: Clean Adaptive Thresholding ---
pts = screen_cnt.reshape(4, 2).astype("float32")
warped = four_point_trans(orig, pts * ratio)

# Convert to grayscale and apply Adaptive Thresholding
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# 251 is the block size (must be odd), 11 is the constant subtracted from mean
T = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                          cv2.THRESH_BINARY, 251, 11)

cv2.imshow("Scanned Perspective", imutils.resize(warped, height=650))
cv2.imshow("Clean Scan (Adaptive)", imutils.resize(T, height=650))
cv2.waitKey(0)