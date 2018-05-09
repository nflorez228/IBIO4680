# OpticalFlow

In Computer Vision the optical flow is the approach of determinig the velocities of objects within frames of a video. Its used in video stream in static cameras in motion-based object detection and tracking systems.
A tool that allows us to interactively work with video frames is OpenCV. This is a free artificial vision library developed by the Intel Corporation in 1999. It was designed primarily to perform applications in real time with high computational efficiency.
Optical Flow takes into account two assumptions:
1.	The intensity of the pixels of an object does not change between consecutive frames.
2.	The neighboring pixels have a similar movement.
With these assumptions we can represent the movement of a pixel taking into account the variations that exist in x, y and time t:
I(x,y,t) = I(x+dx,y+dy,t+dt)
Applying approximations by Taylor series on the right side of the equation, we can find the position gradients for the movement in x as in y. From these gradients the Lucas-Kanade method was applied for developing our algorithm of movement detection to the left or right.

## Problem
Task A
Modify the lk_track.py example, such that you can capture an objetc with your laptop's webcam and detect when it moves to the left or right:

Tips:

Detect the event (sudden increase in lateral flow)
React to the event (you can be creative here)
Give a proper feedback to the user (at the very least a readable console output)
Block other events for a couple of a seconds


## How we've done it

def detectDirection(self,tracks):
        movimiento=np.array([t[-1][0]-t[0][0] for t in tracks])
        promedio = np.mean(movimiento[np.abs(movimiento)>10])
        c=clock()
        if c - self.last_event < self.min_interval:
            return
        im=cv2.imread('cent.png')
        im_resized = cv2.resize(im, (500, 500), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('mov', im_resized)
        if promedio < -10:
            print("Derecha")
            im=cv2.imread('der.png')
            im_resized = cv2.resize(im, (500, 500), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('mov', im_resized)
            self.last_event=c
            return
        if promedio > 10:
            print("Izquierda")
            im=cv2.imread('izq.png')
            im_resized = cv2.resize(im, (500, 500), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('mov', im_resized)
            self.last_event=c
            return

In the last code we check the movement at X axis that comes from the calculated tracks by lk_track.py algorithm itself. With the movement we calculated the average to get were the general movement of the points of interest(POI) goes. 
Next of that we check the clock to get the time for the blocking time of other events to make a threshold in time for detecting the movement.
Then with the time we calculated if the desired time threshold has arrived in order to calculate movement or keep on waiting to reach the threshold.

Next we make the load of the image of the Center movement and resize it to 500*500px to show the image.

Finally we calculate if the movement in the X axis calculated from the first step is greater than a movement treshold (10 for the example) or less than the negative threshold. it means that the movement is grater than 10 the general movements of the POI is to the LEFT, in the case of less than -10 the movement is to the RIGHT. In each case we print in console the movement in spanish, and show an alusive image of the direction of the movement.


## Results
![](https://media3.giphy.com/media/d2lcHJTG5Tscg/200.gif)

## Instructions
run lk_track.y in the scripts folder:
python lk_track.py
 
 Make sure that in the same folder are the images der.png, izq.png, cent.png
 
## References
[1] https://la.mathworks.com/discovery/optical-flow.html
[2] http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html 

![](https://media3.giphy.com/media/d2lcHJTG5Tscg/200.gif)