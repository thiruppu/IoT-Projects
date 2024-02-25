import numpy as np
import time
import cv2
import os
import PySimpleGUI as sg


y_path = r'yolo-coco'

sg.theme('LightGreen')

gui_confidence = .5     # initial settings
gui_threshold = .3      # initial settings
camera_number = 0       # if you have more than 1 camera, change this variable to choose which is used


labelsPath = os.path.sep.join([y_path, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
weightsPath = os.path.sep.join([y_path, "yolov3.weights"])
configPath = os.path.sep.join([y_path, "yolov3.cfg"])
sg.popup_quick_message('Loading YOLO weights from disk.... one moment...', background_color='red', text_color='white')
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
W, H = None, None
win_started = False
cap = cv2.VideoCapture(camera_number)  # initialize the capture device
while True:

    grabbed, frame = cap.read()


    if not grabbed:
        break


    if not W or not H:
        (H, W) = frame.shape[:2]


    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()


    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:

        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]


            if confidence > gui_confidence:

                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)


    idxs = cv2.dnn.NMSBoxes(boxes, confidences, gui_confidence, gui_threshold)


    if len(idxs) > 0:
        for i in idxs.flatten():

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                       confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    # ---------------------------- THE GUI ----------------------------
    if not win_started:
        win_started = True
        layout = [
            [sg.Text('Object Detection', size=(30, 1))],
            [sg.Image(data=imgbytes, key='_IMAGE_')],

            [sg.Exit()]
        ]
        win = sg.Window('Object Detection Webcam ', layout, default_element_size=(14, 1), text_justification='right', auto_size_text=False, finalize=True)
        image_elem = win['_IMAGE_']
    else:
        image_elem.update(data=imgbytes)

    event, values = win.read(timeout=0)
    if event is None or event == 'Exit':
        break


print("[INFO] cleaning up...")
win.close()
