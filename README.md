# HandGestureRecognition
This is a python based Image Recognition project. I have used OpenCV in order to recognize various gestures of hand.

###########################################################################################
This project can be divided into three major portions
1. Background identification
   - Identification is done by calcutating absolute dierence between the frames 
   - Then calculating the accumulated average of these diferences
   
2. Segmenting the hand from the selected ROI using thresholing.

3. Identifying the hand gestures using the following steps.
   - Draw contours around the thresholded image.
   - Find the contour that has the maximum area to identify the hand.
   - Find convexity hull of the hand contour.
   - Calculate the number of deffect points in the convex hull.
   - Depending upon the number of deffect points determine the typ of hand gesture.
     - Number of defects = 0 => Printing "ONE" on the screen
     - Number of defects = 1 => Printing "TWO" on the screen
     - Number of defects = 2 => Printing "THREE" on the screen
     - Number of defects = 3 => Printing "FOUR" on the screen
     - Number of defects = 4 => Printing "FIVE" on the screen
