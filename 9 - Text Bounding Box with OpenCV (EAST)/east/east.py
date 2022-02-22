# =============================================================================
# Script for helper function to use the EAST model with OpenCV
# Template from PyImageSearch
# =============================================================================


# Importing libraries

import numpy as np

# define the two last output layer (prababilities and bounding box coordinates)

EAST_OUTPUT_LAYERS = ["feature_fusion/Conv_7/Sigmoid",
                      "feature_fusion/concat_3"]

## Defining the decode prediction function
# This function requires two arguments and an optional argument
# Score (the value of a prediction for a bounding box)
# Geometry (Allows us to derive the bounding box coordinates)
# Confidence (you can choose the minimal value of the confidence)

def decode_predictions(scores, geometry, minConf = 0.5):
    
    """Decode predictions function.
    This function post-processing the predictions from the EAST model using OpenCV.
    Args:
        scores: array with the predicted confidences
        geometry: array with the geometrics informations about the predicted bounding boxes
        minConf: float variable to determine the minimal confidence score accepted.
    return (rects, confidences)
    """
    
    # grabbing the number of rows and columns from the score vector
    
    (numRows, numCols) = scores.shape[2:4]
    rects = [] #empty list to store the rectangles
    confidences = [] # empty list to store the confidences according with the rectangles
    
    ## loop over the rows
    #Extracting the score and the potential bounding boxes
    for y in range(0, numRows):
        
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        
        ##Loop over the number of columns and filtering the low confidences
        
        for x in range(0, numCols):
            
            #grabbing the confidences
            score = float(scoresData[x])
            
            # condition for the confidence
            
            if score < minConf:
                continue
            
            #computing the offset factor to rescale the bounding box in the image
            
            (offsetX, offsetY) = (x * 4, y * 4)
            
            #grabbing the rotate angle and computing the sine and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            
            #calculating the height and width
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            
            #using the offset and the rotated angle to obtain the rotated bounding box
            offset = ([offsetX + (cos * xData1[x]) + (sin * xData2[x]),
                       offsetY - (sin * xData1[x]) + (cos * xData2[x])])
            
            #deriving the top and bottom right corner
            topLeft = ((-sin * h) + offset[0], (-cos * h) + offset[1])
            topRight = ((-cos * w) + offset[0], (sin * w) + offset[1])
            
            #computing the centers (x, y) of the bounding box
            cX = 0.5 * (topLeft[0] + topRight[0])
            cY = 0.5 * (topLeft[1] + topRight[1])
            
            # our rotated bounding box information consists of the
            # center (x, y)-coordinates of the box, the width and
            # height of the box, as well as the rotation angle
            box = ((cX, cY), (w, h), -1 * angle * 180.0 / np.pi)
            
            #adding the bounding boxes and confidences into the list
            rects.append(box)
            confidences.append(score)
    
    #returning rects and confidences in a tuple format
    return (rects, confidences)

            
   
    