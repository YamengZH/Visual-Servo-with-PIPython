from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from pipython import GCSDevice, pitools
import numpy as np
import imutils
import cv2
import threading, time


class hexapodThread(threading.Thread):
    def __init__(self, name=None, objective=None):
        threading.Thread.__init__(self, name=name)
        self.objective = objective
    def run(self):
        activateHexapod(self.objective)

class cameraThread(threading.Thread):

    def __init__(self, name=None):
        threading.Thread.__init__(self, name=name)

    def run(self):

        cap = cv2.VideoCapture(0)

        while cap.isOpened():

            ret, image = cap.read()

            First_Obj, Second_Obj, object, maxdist, target, target_dist, target_coordinate = parameterConfig()

            gray, maxLoc, cnts = processImage(image)

            if len(cnts) > 1:

                (cnts, _) = contours.sort_contours(cnts)

                for c in cnts:
                    if cv2.contourArea(c) < 50:
                        continue
                    box = cv2.minAreaRect(c)
                    box = cv2.boxPoints(box)
                    box = np.array(box, dtype="int")
                    box = perspective.order_points(box)
                    cX = int(np.average(box[:, 0]))
                    cY = int(np.average(box[:, 1]))
                    object.append(np.vstack([box, (cX, cY)]))

                if len(object) > 1:
                    for count in range(len(object)):
                        maxdist.append(dist.euclidean(object[count][-1], maxLoc))

                    First_Obj = object[np.argmin(maxdist)]

                    for count in range(len(object)):
                        if count != np.argmin(maxdist):
                            if gray[int(object[count][-1][1])][int(object[count][-1][0])] < 180:
                                target.append(object[count])

                    if len(target):
                        for count in range(len(target)):
                            target_dist.append(dist.euclidean(target[count][-1], First_Obj[-1]))

                        Second_Obj = target[np.argmin(target_dist)]

                        cv2.drawContours(image, [First_Obj[0:4].astype("int")], -1, (0, 255, 0), 2)
                        cv2.drawContours(image, [Second_Obj[0:4].astype("int")], -1, (0, 255, 0), 2)
                        ((xA, yA), (xB, yB), color) = (First_Obj[-1], Second_Obj[-1], (255, 0, 255))
                        # print((int(xA), int(yA)), (int(xB), int(yB)))
                        cv2.circle(image, (int(xA), int(yA)), 2, color, -1)
                        cv2.circle(image, (int(xB), int(yB)), 2, color, -1)
                        cv2.arrowedLine(image, (int(xA), int(yA)), (int(xB), int(yB)), color, 1)
                        D = dist.euclidean((xA, yA), (xB, yB))
                        # print(D)
                        (mX, mY) = midpoint((xA, yA), (xB, yB))
                        cv2.putText(image, "A", (int(xA + 5), int(yA + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                        cv2.putText(image, "B", (int(xB + 5), int(yB + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                        cv2.putText(image, "{:.1f}pix".format(D), (int(mX), int(mY)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            cv2.imshow("Image", image)

            if cv2.waitKey(1) & 0xff == ord(' '):
                break

        cap.release()
        cv2.destroyAllWindows()

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def processImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    edged = cv2.Canny(gray, 50, 100)

    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    return gray, maxLoc, cnts

def parameterConfig():
    First_Obj = None
    Second_Obj = None
    object = []
    maxdist = []
    target = []
    target_dist = []
    target_coordinate = []
    return First_Obj, Second_Obj, object, maxdist, target, target_dist, target_coordinate

def hexapodConfig(pidevice):
    pidevice.ConnectRS232(comport=5, baudrate=115200)
    # print('connected: {}'.format(pidevice.qIDN().strip()))
    pitools.startup(pidevice, stages=STAGES, refmodes=REFMODES)
    # rangemin = pidevice.qTMN()
    # rangemax = pidevice.qTMX()
    # curpos = pidevice.qPOS()
    # return rangemin, rangemax, curpos

def hexapodInitialize(pidevice):
    startpos = [0] * len(pidevice.axes)
    pidevice.MOV(axes=pidevice.axes, values=startpos)
    pitools.waitontarget(pidevice, pidevice.axes)
    # curpos = pidevice.qPOS()
    # print(curpos)

def hexapodMove(pidevice, objective):
    pidevice.MOV(axes=pidevice.axes, values=objective)
    pitools.waitontarget(pidevice, pidevice.axes)
    # position = pidevice.qPOS()
    # print(position)
    # print('Complete!')

def activateHexapod(objective):
    with GCSDevice(CONTROLLERNAME) as pidevice:
        hexapodConfig(pidevice) # connect with RS232 and return features.
        for obj in objective:
            pos = (obj[0], obj[1])
            pidevice.MOV(axes=pidevice.axes[:2], values=pos)
            pitools.waitontarget(pidevice, pidevice.axes[:2])

def hexapodZero():
    with GCSDevice(CONTROLLERNAME) as pidevice:
        hexapodConfig(pidevice)
        hexapodInitialize(pidevice)

def hexapodX_10():
    with GCSDevice(CONTROLLERNAME) as pidevice:
        hexapodConfig(pidevice)
        hexapodMove(pidevice, [10, 0] + [0] * 4)

def hexapodY_10():
    with GCSDevice(CONTROLLERNAME) as pidevice:
        hexapodConfig(pidevice)
        hexapodMove(pidevice, [0, 10] + [0] * 4)

def calJacobian():
    hexapodZero()
    # time.sleep(0.5)
    laserPoint, darkPoint = acquireObject()
    hexapodX_10()
    # time.sleep(0.5)
    laserPointX, darkPointX = acquireObject()
    hexapodZero()
    hexapodY_10()
    # time.sleep(0.5)
    laserPointY, darkPointY = acquireObject()

    deltaX = np.mean(np.subtract(darkPointX, darkPoint), axis=0)
    deltaY = np.mean(np.subtract(darkPointY, darkPoint), axis=0)

    theta = np.arctan(np.divide(np.mean([-deltaX[1], deltaY[0]]), np.mean([deltaX[0], deltaY[1]])))

    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    Jacobian = np.dot(rot_mat, 10 * np.reciprocal(np.mean([deltaX[0], deltaY[1]]) * np.cos(theta) \
                                             + np.mean([-deltaX[1], deltaY[0]]) * np.sin(theta)))

    print(laserPoint)

    print(darkPoint)

    print(np.subtract(laserPoint * len(darkPoint), darkPoint))

    objective = np.transpose(np.dot(Jacobian, np.transpose(np.subtract(laserPoint * len(darkPoint), darkPoint))))

    print(objective)

    hexapodZero()

    return objective

CONTROLLERNAME = 'C-887'
STAGES = None
REFMODES = 'FRF'

def acquireObject():
    global isObjectCaptured
    isObjectCaptured = False
    laserPoint = []
    darkPoint = []

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while cap.isOpened():
        ret, image = cap.read()
        First_Obj, Second_Obj, object, maxdist, target, target_dist, target_coordinate = parameterConfig()
        gray, maxLoc, cnts = processImage(image)
        if len(cnts) > 1:
            (cnts, _) = contours.sort_contours(cnts)
            for c in cnts:
                if cv2.contourArea(c) < 50:
                    continue
                box = cv2.minAreaRect(c)
                box = cv2.boxPoints(box)
                box = np.array(box, dtype="int")
                box = perspective.order_points(box)
                cX = int(np.average(box[:, 0]))
                cY = int(np.average(box[:, 1]))
                object.append(np.vstack([box, (cX, cY)]))
            if len(object) > 1:
                for count in range(len(object)):
                    maxdist.append(dist.euclidean(object[count][-1], maxLoc))
                First_Obj = object[np.argmin(maxdist)]
                for count in range(len(object)):
                    if count != np.argmin(maxdist):
                        if gray[int(object[count][-1][1])][int(object[count][-1][0])] < 180:
                            target.append(object[count])
                if len(target):
                    for count in range(len(target)):
                        target_dist.append(dist.euclidean(target[count][-1], First_Obj[-1]))
                        target_coordinate.append(target[count][-1][0])
                    laserPoint.append(First_Obj[-1])
                    sorted_indices = np.argsort(target_coordinate)
                    for count in sorted_indices:
                        darkPoint.append(target[count][-1])
                    isObjectCaptured = True
        if isObjectCaptured:
            break
    cap.release()
    cv2.destroyAllWindows()
    return laserPoint, darkPoint

def main():

    objective = calJacobian()

    camera = cameraThread(name='cameraThread')

    hexapod = hexapodThread(name='hexapodThread', objective=objective)

    camera.start()

    hexapod.start()

if __name__ == '__main__':
    main()
