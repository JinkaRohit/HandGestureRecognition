# organize imports
import cv2
import math

# global variables
bg = None
cmp_scr = 0
usr_scr = 0
round = 0

#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def runAvg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)


#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff_image = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff_image, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 100, 350, 350, 590

    # initialize num of frames
    num_frames = 0

    # keep looping, until interrupted
    while(True):
        # capturing the current frame
        (_, frame) = camera.read()

        # flipping the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clonning the frame
        clone = frame.copy()

        # get the height and width(shape) of the frame
        (height, width) = frame.shape[:2]

        # get the RegionOfInterest(ROI)
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            runAvg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)

                # Find contour with maximum area
                contour = segmented

                # Find convex hull
                hull = cv2.convexHull(contour)


                # Find convexity defects
                hull = cv2.convexHull(contour, returnPoints=False)
                defects = cv2.convexityDefects(contour, hull)

                # Use cosine rule to find angle of the far point from the start and end point
                # i.e. the convex points (the finger tips) for all defects
                count_defects = 0

                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])

                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

                    # if angle > 90 draw a circle at the far point
                    if angle <= 90:
                        count_defects += 1
                        cv2.circle(roi, far, 1, [0, 0, 255], -1)

                    cv2.line(roi, start, end, [0, 255, 0], 2)

                # Print number of fingers
                if count_defects == 0:
                    cv2.putText(clone, "ONE", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                elif count_defects == 1:
                    cv2.putText(clone, "TWO", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                elif count_defects == 2:
                    cv2.putText(clone, "THREE", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                elif count_defects == 4:
                    cv2.putText(clone, "FIVE", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                elif count_defects == 3:
                    cv2.putText(clone, "FOUR", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                else:
                    pass


        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()

