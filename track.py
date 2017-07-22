#!/usr/local/bin/python
"""
Created on Fri Sep  25 19:42:41 2015
Name:    track.py
Purpose: Track Drosphila larvae and analyze bahavior.
Author:  Andrea Vaccari (av9g@virginia.edu)

If you use our software, please cite our work:
    <insert citation here>

MIT License

Copyright (c) 2017 Andrea Vaccari

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
TODO:
- verify if y is the best direction to account (maybe projection on direction
  perpendicular to best linear fit to top left corners of selected larvae)
- evaluate movement along that direction
- smooth out noise
- detect max and min
- evaluate each period
- evaluate delays within period
- evaluate average period
- evaluate displacements for each period
- evaluate average displacement
- cross-correlation could be used to quantify phase difference between tracked
  larvae motion
- can we measure contraction of larvae if the user select both head and tail?
- can we correlate motion of larvae with the nearby meniscus? Can we image
  food+larvae vs air? (volume of meniscus vs time?)
"""

import cv2
import argparse
import numpy as np
import Tkinter as tk
import tkFileDialog as tkfd
import tkMessageBox as tkmb
import matplotlib.pyplot as plt
from os.path import splitext, basename
import csv


def sorteva(eva):
    srt_idx = np.argsort(np.abs(eva))
    l1 = np.reshape(eva[srt_idx == 0], np.shape(eva)[:2])
    l2 = np.reshape(eva[srt_idx == 1], np.shape(eva)[:2])

    return np.dstack((l1, l2))


def tubularity(eva):
    seva = sorteva(eva)
    l1 = seva[:, :, 0]
    l2 = seva[:, :, 1]

    Rb = l1 / l2

    S2 = l1 * l1 + l2 * l2

    mx = 0.2 * np.max(np.sum(eva, 2))
    c = -0.5 / (mx * mx)

    v = np.exp(-2.0 * Rb * Rb) * (1.0 - np.exp(c * S2))
    v[~np.isfinite(v * l2)] = 0.0
    v[l2 >= 0] = 0.0

    return v


def hessian(array):
    (dy, dx) = np.gradient(array)
    (dydy, dxdy) = np.gradient(dy)
    (dydx, dxdx) = np.gradient(dx)
    return np.dstack((dxdx, dydx, dxdy, dydy))


def eval2ds(stack):
    a = stack[:, :, 0]
    b = stack[:, :, 1]
    c = stack[:, :, 2]
    d = stack[:, :, 3]

    T = a + d
    D = a * d - b * c

    Th = 0.5 * T
    T2 = T * T
    C = np.sqrt(T2 / 4.0 - D)

    return np.dstack((Th + C, Th - C))


def tubes(img, sigma_rng):
    img = img.astype(np.float32) / 255.0

    s = np.arange(sigma_rng[0], sigma_rng[1], 2)

    stk = np.dstack([np.empty_like(img)] * len(s))
    for i in range(len(s)):
        blr = cv2.GaussianBlur(img, (s[i], s[i]), 0)
        hes = hessian(blr)
        eva = eval2ds(hes)
        stk[:, :, i] = tubularity(eva)
    tub = 255.0 * np.max(stk, 2)

    return tub.astype(np.uint8)



class trackedArea(object):
    def __init__(self, corners):
        if corners[0][0] < corners[1][0]:
            self.c = corners[0][0]
        else:
            self.c = corners[1][0]

        if corners[0][1] < corners[1][1]:
            self.r = corners[0][1]
        else:
            self.r = corners[1][1]

        self.w = np.abs(corners[0][0] - corners[1][0])
        self.h = np.abs(corners[0][1] - corners[1][1])

        corn = []
        corn.append((self.c, self.r))
        corn.append((self.c + self.w, self.r + self.h))
        self.corners = np.asarray(corn)

        self.initLoc = np.array((self.c, self.r))
        self.location = np.array((self.c, self.r))
        self.deltaLoc = None

        self.templ = None
        self.templStack = None
        self.templCnt = None
        self.templWeights = None
        self.stackSize = None


    def initKalman(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], dtype=np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], dtype=np.float32)
        self.kf.processNoiseCov = 1.0 * np.eye(4, dtype=np.float32)
        self.kf.measurementNoiseCov = 1.0 * np.eye(2, dtype=np.float32)
        post = np.array([self.c, self.r, 0, 0], dtype=np.float32)
        post.shape = (4, 1)
        self.kf.statePost = post

    def getKalmanPredict(self):
        pred = self.kf.predict()
        pred = pred[:2].astype(np.int)
        return tuple((pred[0][0], pred[1][0]))

    def setKalmanCorrect(self, loc):
        loc = np.array(loc, dtype=np.float32)
        loc.shape = (2, 1)
        self.kf.correct(loc)

    def setStackSize(self, size):
        self.stackSize = size

    def getCorners(self):
        return [tuple(self.corners[0]), tuple(self.corners[1])]

    def getHalfCorners(self):
        corn = self.corners - (self.w/2, self.h/2)
        return [tuple(corn[0]), tuple(corn[1])]

    def getEnlargedCorners(self, pxls):
        corn = self.corners - (self.w/2, self.h/2)
        corn += np.asarray([[-pxls, -pxls], [pxls, pxls]])
        return [tuple(corn[0]), tuple(corn[1])]

    def getcrwh(self):
        return (self.c, self.r, self.w, self.h)

    def setcrwh(self, window):
        self.c = window[0]
        self.r = window[1]
        self.w = window[2]
        self.h = window[3]
        self.setLocation((self.c, self.r))

    def updateWindow(self, loc):
        self.c = loc[0]
        self.r = loc[1]
        self.setLocation((self.c, self.r))
        corn = []
        corn.append((self.c, self.r))
        corn.append((self.c + self.w, self.r + self.h))
        self.corners = np.asarray(corn)

    def setLocation(self, loc):
        delta = np.asarray(loc) - self.initLoc
        self.deltaLoc = np.sqrt(np.inner(delta, delta))
        self.location = np.asarray(loc)

    def getLocation(self):
        return self.location

    def getInitLoc(self):
        return self.initLoc

    def getDeltaLoc(self):
        return self.deltaLoc

    def setTemplate(self, image):
        self.templ = image[self.r:self.r+self.h, self.c:self.c+self.w].copy()

        if self.templCnt is None:
            self.templCnt = 0
            self.templStack = np.concatenate([self.templ[..., np.newaxis] for i in range(self.stackSize)], axis=3)

        self.templStack[:, :, :, self.templCnt] = self.templ
        self.templCnt += 1
        self.templCnt %= self.stackSize  # If 30, reset to 0

    def getGrayStackAve(self):
        ave = self.getStackAve()
        return cv2.cvtColor(ave, cv2.COLOR_BGR2GRAY)

    def getStack(self):
        stack = np.concatenate([self.templStack[..., i] for i in range(self.stackSize)], axis=1)
        return np.concatenate((stack, np.zeros_like(self.templ), self.getStackAve()), axis=1)

    def getStackAve(self):
        self.templWeights = 0.25 * np.ones(self.stackSize)
        lastTemplCnt = self.templCnt - 1
        if self.templCnt == 0:
            lastTemplCnt = self.stackSize - 1
        self.templWeights[lastTemplCnt] = 1.0
        ave = np.average(self.templStack, axis=3, weights=self.templWeights).astype(np.uint8)
        return ave

    def getGrayTemplate(self):
        return cv2.cvtColor(self.templ, cv2.COLOR_BGR2GRAY)

    def dist(self, other):
        diff = self.location - other.location
        return np.sqrt(np.inner(diff, diff))



class watch(object):
    def __init__(self, vid):
        if vid is None:
            root = tk.Tk()
            root.withdraw()
            root.update()
            root.iconify()
            vid = tkfd.askopenfilename()

        if vid is '':
            raise IOError
        else:
            self.vid = vid

        self.showHelp('instructions')

        self.cap = cv2.VideoCapture(self.vid)

        self.enhance = False

        self.sourceFrame = None
        self.processedFrame = None
        self.workingFrame = None
        self.undoFrames = []
        self.frameNo = 0
        self.lastFrame = False
        self.userInteraction = False

        self.mainWindow = basename(self.vid)
        cv2.namedWindow(self.mainWindow)

        self.selectionWindow = 'Select areas to track'

        self.refPt = []
        self.tracking = False
        self.trackedAreasList = []
        self.showMatch = False
        self.showTemplate = False
        self.trackDump = None

        self.pause = False

        self.distPlot = None
        self.plotLength = 600
        self.driftMax = 1
        self.distMax = 1

        plt.ion()




    def showHelp(self, menu):
        if menu == 'main':
            message = "Active keys:\n" + \
                      "'a' -> selects areas to track (Click, drag, release)\n" + \
                      "'e' -> toggle video enhancement\n" + \
                      "'p' -> pause/run the video\n" + \
                      "'t' -> toggles the display of the current template\n" + \
                      "'m' -> toggles the display of the current matching\n" + \
                      "'h' -> shows this help\n" + \
                      "\n'q' -> quits"
        elif menu == 'select':
            message = "To select an area, click, drag, and release.\n" + \
                      "Select area keys:\n" + \
                      "'l' -> clear last selection\n" + \
                      "'c' -> clear all selections\n" + \
                      "'t' -> start tracking\n" + \
                      "'h' -> shows this help\n" + \
                      "\n'q' -> quits selection"
        elif menu == 'instructions':
            message = "The video will start paused. Push 'p' to run." + \
                      "To perform an analysis:\n" + \
                      "1. Push 'a' to switch to selection mode\n" + \
                      "2. Select the end section of the larva at the center.\n" + \
                      "3. Select the end sections of the larvae to the left and the right.\n" + \
                      "\n'h' -> context specific help"
        else:
            message = "You shouldn't be here!"

        tkmb.showinfo('Help',
                      message=message,
                      icon=tkmb.INFO)



    def mouseInteraction(self, event, x, y, flags, params):
        if self.userInteraction is True:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.refPt = [(x, y)]
                self.workingFrame[y, x] = [0, 0, 255]
                self.showFrame(self.selectionWindow, self.workingFrame)
            elif event == cv2.EVENT_LBUTTONUP:
                self.undoFrames.append(self.workingFrame.copy())
                self.refPt.append((x, y))
                if self.refPt[0][0] != self.refPt[1][0] and self.refPt[0][1] != self.refPt[1][1]:
                    area = trackedArea(self.refPt)
                    area.setStackSize(30)
                    area.setTemplate(self.processedFrame)
#                    area.initKalman()
                    corn = area.getCorners()
                    self.trackedAreasList.append(area)

                    cv2.rectangle(self.workingFrame,
                                  corn[0], corn[1],
                                  (0, 0, 255), 1)

                    self.showFrame(self.selectionWindow, self.workingFrame)




    def selectArea(self):
        self.userInteraction = True
        cv2.namedWindow(self.selectionWindow)
        cv2.setMouseCallback(self.selectionWindow, self.mouseInteraction)
        self.workingFrame = self.processedFrame.copy()
        self.showFrame(self.selectionWindow, self.workingFrame)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self.undoFrames = []
                break
            elif key == ord('c'):
                self.workingFrame = self.processedFrame.copy()
                self.trackedAreasList = []
                self.undoFrames = []
                self.showFrame(self.selectionWindow, self.workingFrame)
            elif key == ord('l'):
                try:
                    self.trackedAreasList.pop()
                except IndexError:
                    pass
                else:
                    self.workingFrame = self.undoFrames.pop()
                    self.showFrame(self.selectionWindow, self.workingFrame)
            elif key == ord('t'):
                self.undoFrames = []
                self.trackArea = self.refPt
                self.tracking = True
                self.trackDump = []
                if self.pause is True:
                    self.pause = False
                break
            elif key == ord('h'):
                self.showHelp('select')

        cv2.destroyWindow(self.selectionWindow)
        self.userInteration = False




    def readFrame(self):
        ret, frame = self.cap.read()
        if ret == 0:
            self.cap.release()
            self.lastFrame = True
        else:
            self.frameNo += 1
            self.sourceFrame = frame




    def trackObjects(self):
        for area in self.trackedAreasList:
            # Template matching
            gray = cv2.cvtColor(self.processedFrame, cv2.COLOR_BGR2GRAY)
            templ = area.getGrayStackAve()
            cc = cv2.matchTemplate(gray, templ, cv2.TM_CCOEFF_NORMED)
            cc = cc * cc * cc * cc
            _, cc = cv2.threshold(cc, 0.1, 0, cv2.THRESH_TOZERO)
            cc8 = cv2.normalize(cc, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            mask = np.zeros_like(cc8)

            # Search match within template region
            mcorn = area.getEnlargedCorners(0) # If not 0, enalrge the search
            cv2.rectangle(mask, mcorn[0], mcorn[1], 255, -1)
            _, _, _, mx = cv2.minMaxLoc(cc8, mask)

#            kp = area.getKalmanPredict()
#            area.updateWindow(kp)
#            area.setTemplate(self.processedFrame)

            # Prevent large spatial jumps
            (c, r, _, _) = area.getcrwh()
            jump = 10
            if abs(c - mx[0]) < jump and abs(r - mx[1]) < jump:
#                area.setKalmanCorrect(mx)
                area.updateWindow(mx)
            else:
#                area.setKalmanCorrect((c, r))
                area.updateWindow((c, r))
            area.setTemplate(self.processedFrame)

            # Show the template stack
            if self.showTemplate is True:
                cv2.imshow('Stack: '+str(area), area.getStack())
            else:
                try:
                    cv2.destroyWindow('Stack: '+str(area))
                except:
                    pass

            # Show the matching results
            if self.showMatch is True:
                cv2.rectangle(cc8, mcorn[0], mcorn[1], 255, 1)
                cv2.circle(cc8, mx, 5, 255, 1)
                cv2.imshow('Match: '+str(area), cc8)
            else:
                try:
                    cv2.destroyWindow('Match: '+str(area))
                except:
                    pass

            # Draw the tracked area on the image
            corn = area.getCorners()
            cv2.rectangle(self.workingFrame,
                          corn[0], corn[1],
                          (0, 255, 0), 1)

#            self.showFrame()
#            raw_input('wait')


    def showBehavior(self):
        idx = self.frameNo % self.plotLength

        dump = [self.frameNo]  # Store line for CSV file

        # Draw drift from original position
        drift = []
        for area in self.trackedAreasList:
            drift.append(area.getDeltaLoc())
            start = area.getInitLoc()
            stop = area.getLocation()
            dump.extend(stop)
            cv2.line(self.workingFrame, tuple(start), tuple(stop), (255,0, 0))
        dump.extend(drift)  # Add (x, y) drift to CSV
        drift = np.asarray(drift)

        # Create and update drift plot
        for i in range(len(drift)):
            try:
                plt.subplot(2, 1, 1)
                data = self.driftPlot[i].get_ydata()
            except (AttributeError, TypeError):
                x = np.arange(0, self.plotLength)
                y = np.zeros((self.plotLength, len(drift)))
                self.driftPlot = plt.plot(x, y)
                plt.title('Drift from starting position')
                plt.ylabel('Drift (pixels)')
                plt.tick_params(axis='x', labelbottom='off')
            else:
                data[idx] = drift[i]
                self.driftPlot[i].set_ydata(data)

        if drift.max() > self.driftMax:
            self.driftMax = drift.max()
        plt.ylim([0, self.driftMax])

        # Draw distance from first selected larva
        dist = []
        mainArea = self.trackedAreasList[0]
        stop = tuple(mainArea.getLocation())
        for area in self.trackedAreasList:
            dist.append(area.dist(mainArea))
            start = tuple(area.getLocation())
            cv2.line(self.workingFrame, start, stop, (255,0, 0))
        dump.extend(dist)  # Add distance to CSV
        dist = np.asarray(dist)

        # Create and update distane plot
        for i in range(len(dist)):
            try:
                plt.subplot(2, 1, 2)
                data = self.distPlot[i].get_ydata()
            except (AttributeError, TypeError):
                x = np.arange(0, self.plotLength)
                y = np.zeros((self.plotLength, len(dist)))
                self.distPlot = plt.plot(x, y)
                plt.title('Distance from main larva')
                plt.xlabel('Frame')
                plt.ylabel('Distance (pixels)')
            else:
                data[idx] = dist[i]
                self.distPlot[i].set_ydata(data)

        if dist.max() > self.distMax:
            self.distMax = dist.max()
        plt.ylim([0, self.distMax])

        plt.show()
        plt.draw()
        plt.pause(0.001)

        # Add to track dump for CSV file
        self.trackDump.append(dump)






    def processFrame(self):
        # If we are enhancing the image
        if self.enhance:
            # Frangi vesselness to highlight tubuar structures
            gray = cv2.cvtColor(self.sourceFrame, cv2.COLOR_BGR2GRAY)
            tub = tubes(gray, [5, 12])
            tubular = cv2.cvtColor(tub, cv2.COLOR_GRAY2BGR)

            # Merge with original to ennhance tubular structures
            high = 0.3
            rest = 1.0 - high
            colorized = cv2.addWeighted(self.sourceFrame, rest, tubular, high, 0.0)
    #        colorized = cv2.add(self.sourceFrame, tubular)

            # Tile horizontally
            self.processedFrame = np.concatenate((self.sourceFrame,
                                                  tubular,
                                                  colorized),
                                                 axis=1)
        else:
            self.processedFrame = self.sourceFrame;

        self.workingFrame = self.processedFrame.copy()

        # If we are tracking, track and show analysis
        if self.tracking is True:
            self.trackObjects()
            self.showBehavior()


    def showFrame(self, window, frame):
        cv2.imshow(window, frame)




    def watch(self):
        while self.lastFrame is False:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('t'):
                self.showTemplate = not self.showTemplate
            elif key == ord('m'):
                self.showMatch = not self.showMatch
            elif key == ord('e'):
                self.enhance = not self.enhance
                self.processFrame()
                self.showFrame(self.mainWindow, self.workingFrame)
            elif key == ord('q'):
                break
            elif key == ord('a'):
                self.selectArea()
            elif key == ord('h'):
                self.showHelp('main')
            elif key == ord('p'):
                self.pause = not self.pause
            else:
                if not self.pause:
                    self.readFrame()
                    self.processFrame()
                    self.showFrame(self.mainWindow, self.workingFrame)
                if self.frameNo == 1:
                    self.pause = True

        # If any, dump tracking data to file
        if self.trackDump is not None:
            (fileName, _) = splitext(self.vid)
            no = len(self.trackedAreasList)
            header1 = [['Larva-' + str(i), ''] for i in range(no)]
            header1 = [''] + [i for sub in header1 for i in sub]
            header1.extend(['Larva-' + str(i) for i in range(no)] * 2)
            header2 = ['frame'] + ['x', 'y'] * no + ['drift'] * no + ['distance'] * no
            with open(fileName + '.csv', 'w') as csvFile:
                dump = csv.writer(csvFile)
                dump.writerow(header1)
                dump.writerow(header2)
                dump.writerows(self.trackDump)

        cv2.waitKey(-1)


    def __enter__(self):
        return self



    def __exit__(self, exec_type, exec_value, traceback):
        try:
            self.cap.release()
        except AttributeError:
            pass
        cv2.destroyAllWindows()
        plt.close('all')




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file",
                        help="File to analyze.")

    args = parser.parse_args()

    again = True

    while again is True:
        try:
            with watch(args.file) as w:
                w.watch()
        except IOError:
            pass

        # Do you want to analyze another file?
        again = tkmb.askyesno("Analyze another?",
                              "Do you want to open another file?")
