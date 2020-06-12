# Sections of code taken from yolo.py, stereo_to_3d.py and stereo_disparity.py
# https://github.com/tobybreckon/stereo-disparity
# https://github.com/tobybreckon/python-examples-cv

import cv2
import numpy as np
import os
import math

master_path_to_dataset = "data"  # directory containing images
directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"

pause_playback = False  # pause until key press after each image

full_path_directory_left = os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right = os.path.join(master_path_to_dataset, directory_to_cycle_right);

left_file_list = sorted(os.listdir(full_path_directory_left))

# yolo.py setup
################################################################################

# Initialize the parameters
confThreshold = 0.56  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

classesFile = "yolo/coco.names"
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolo/yolov3.cfg"
modelWeights = "yolo/yolov3.weights"

# load configuration and weight files for the model and load the network using them
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# defaults DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib available or DNN_BACKEND_OPENCV otherwise
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# change to cv2.dnn.DNN_TARGET_CPU (slower) if this causes issues (should fail gracefully if OpenCL not available)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

################################################################################

# stereo_to_3d.py setup
################################################################################

max_disparity = 128
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21)

# fixed camera parameters for this stereo setup (from calibration)

camera_focal_length_px = 399.9745178222656  # focal length in pixels
stereo_camera_baseline_m = 0.2090607502  # camera baseline in metres

image_centre_h = 262.0
image_centre_w = 474.5

################################################################################


def project_disparity_to_3d(disparity, left, top, right, bottom):
    points = []

    f = camera_focal_length_px
    B = stereo_camera_baseline_m

    # check if left, top, right and bottom are within image dimensions
    if top < 0:
        top = 0
    if bottom > 543:
        bottom = 543
    if left < 0:
        left = 0
    if right > 1023:
        right = 1023

    for y in range(top, bottom + 1):  # top - bottom of object is the y axis index
        for x in range(left, right + 1):  # left - right of object is the x axis index

            # if we have a valid non-zero disparity
            if disparity[y, x] > 0:

                # calculate corresponding 3D point [X, Y, Z]
                Z = (f * B) / disparity[y, x]

                X = ((x - image_centre_w) * Z) / f
                Y = ((y - image_centre_h) * Z) / f

                # add to points
                points.append([X, Y, Z])
    # if no points found, rerun with greater dimensions
    if not points:
        points = project_disparity_to_3d(disparity, left - 10, top - 10, right + 10, bottom - 10)

    return points


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(classId, left, top, right, bottom, disparity_scaled):
    # convert all points found in object's rectangle to 3d coordinates
    points = project_disparity_to_3d(disparity_scaled, left, top, right, bottom)
    dist = []

    # for each point in 3d, calculate distance from camera - sqrt(x^2 + y^2 + z^2)
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        z = points[i][2]
        dist.append(math.sqrt(math.pow(x, 2) + math.pow(y, 2) + math.pow(z, 2)))

    # sort list and select lower quartile value as distance from camera
    dist.sort()
    dist = dist[int(len(points) / 4)]

    # create 1.5m sphere around camera to represent distance from car
    dist -= 1.5
    label = str(round(dist,1)) + "m"

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s: %s' % (classes[classId], label)

    # Draw a bounding box.
    if classes[classId] == "person":
        cv2.rectangle(frame, (left, top), (right, bottom), (50, 178, 255), 3)
    else:
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                  (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return dist


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, disparity_scaled):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            if classId < 9:  # only objects 0-8 are vehicles and people
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])


    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    closest = 0
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        #   any object containing y values > 500 and x values containing 512 is the car itself and can therefore be ignored
        if top + height > 500 and left < 512 and left + width > 512:
            continue
        dist = drawPred(classIds[i], left, top, left + width, top + height, disparity_scaled)

        # calculate closest object to car
        if closest == 0:
            closest = dist
        elif closest > dist:
            closest = dist
    return closest


for filename_left in left_file_list:

    # from the left image filename get the corresponding right image
    filename_right = filename_left.replace("_L", "_R")
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right)

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)):

        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such

        frame = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images

        grayL = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # perform preprocessing - raise to the power, as this subjectively appears
        # to improve subsequent disparity calculation

        grayL = np.power(grayL, 0.75).astype('uint8')
        grayR = np.power(grayR, 0.75).astype('uint8')

        # compute disparity image from undistorted and rectified stereo images
        # that we have loaded
        # (which for reasons best known to the OpenCV developers is returned scaled by 16)

        disparity = stereoProcessor.compute(grayL, grayR)

        # filter out noise and speckles (adjust parameters as needed)

        dispNoiseFilter = 5  # increase for more agressive filtering
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter)

        # scale the disparity to 8-bit for viewing
        # divide by 16 and convert to 8-bit image (then range of values should
        # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
        # so we fix this also using a initial threshold between 0 and max_disparity
        # as disparity=-1 means no disparity available

        _, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)
        disparity_scaled = (disparity / 16.).astype(np.uint8)

        # display image (scaling it to the full 0->255 range based on the number
        # of disparities in use for the stereo part)

        cv2.imshow("disparity", (disparity_scaled * (256. / max_disparity)).astype(np.uint8));

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        closest = postprocess(frame, outs, disparity_scaled)

        print(filename_left)
        print(filename_right + " : nearest detected scene object (" + str(round(closest, 1)) + "m)")
        print()
        cv2.imshow('left image', frame)

        # keyboard input for exit (as standard), save disparity and cropping
        # exit - x
        # save - s
        # pause - space

        key = cv2.waitKey(40 * (not (pause_playback))) & 0xFF  # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):  # exit
            break
        elif (key == ord('s')):  # save
            cv2.imwrite("left.png", frame)
        elif (key == ord(' ')):  # pause (on next frame)
            pause_playback = not (pause_playback)
    else:
        print("-- files skipped (perhaps one is missing or not PNG)")
        print()
