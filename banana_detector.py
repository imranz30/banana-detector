# USAGE
# python banana_detector.py --image images/b1.jpg

# import the necessary packages
import argparse
import cv2
import numpy as np
import glob

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-c", "--cascade",
	#default="banana_classifier.xml",
	help="path to banana detector haar cascade")
args = vars(ap.parse_args())

# load the input image and convert it to grayscale
image = cv2.imread(args["image"])
image2 = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load the cat detector Haar cascade, then detect cat faces
# in the input image
detector = cv2.CascadeClassifier(args["cascade"])
rects = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(50, 50))

print rects

# loop over the cat faces and draw a rectangle surrounding each
for (i, (x, y, w, h)) in enumerate(rects):
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.putText(image, "Banana".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

# show the detected cat faces
cv2.imshow("Bananas", image)
cv2.waitKey(0)

# generating an edge image for segmentation and bounding area detection
#def auto_canny(image, sigma=0.75):
    # compute the median of the single channel pixel intensities
#    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
 #   lower = int(max(0, (1.0 - sigma) * v))
 #   upper = int(min(255, (1.0 + sigma) * v))
 #   edged = cv2.Canny(image, lower, upper)

    # return the edged image
 #  return edged

#edge_image = auto_canny(image2)
#cv2.imshow("Edged Image", edge_image)
#cv2.waitKey(0)

