import cv2
import numpy as np

cam = cv2.VideoCapture('Lane Detection Test Video 01.mp4')

left_top_x=left_bottom_x=right_top_x=right_bottom_x=left_bottom_y=left_top_y=right_top_y=right_bottom_y=left_top_x_f=left_bottom_x_f=right_top_x_f=right_bottom_x_f=0

while True:

    ret, frame = cam.read()
#Step2  Shrink the frame

    frame = cv2.resize(frame, (480, 280))
    # print(frame.shape)

#Step3  Convert the frame to GrayScale

    frame_o=frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#Step4 Select only the road

    frame2 = np.zeros((frame.shape[0], frame.shape[1]), np.uint8) #creez blank frame

    pts = np.array([[int(frame.shape[1] * 0.45), int(frame.shape[0] * 0.75)],
                    [int(frame.shape[1] * 0.55), int(frame.shape[0] * 0.75)], # coordonate trapez
                    [int(frame.shape[1] * 1), int(frame.shape[0] * 1)],
                    [int(frame.shape[1] * 0), int(frame.shape[0] * 1)]],
                   np.int32)

#Step5 Birds-eye view

    pts_frame = np.array([[0, 0],
                          [frame.shape[1], 0],
                          [frame.shape[1], frame.shape[0]],
                          [0, frame.shape[0]]
                          ],
                         np.int32)

    cv2.fillConvexPoly(frame2, pts, (1, 1, 1))  # am colorat trapezul cu alb

    trapezoid_bounds = np.float32(pts)
    screen_bounds = np.float32(pts_frame)
    magic_matrix = cv2.getPerspectiveTransform(trapezoid_bounds, screen_bounds)
    frame3 = cv2.warpPerspective(frame * frame2, magic_matrix, (frame.shape[1], frame.shape[0]))

#Step6 Add a bit of blur

    frame4=frame3.copy()
    frame4=cv2.blur(frame4,ksize=(3,3))

#Step7 Edge detection


    sobel_vertical=np.float32([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])

    sobel_horizontal=np.transpose(sobel_vertical)
    frame5=np.float32(frame4)
    frame6=np.float32(frame4)
    frame5=cv2.filter2D(frame5,-1,sobel_vertical)
    frame6=cv2.filter2D(frame6,-1,sobel_horizontal)
    frameV=frame5
    frameH=frame6
    frame5=np.power(frame5,2)
    frame6=np.power(frame6,2)
    M=np.add(frame5,frame6)
    M=np.sqrt(M)
    M= cv2.convertScaleAbs(M)

#Step8 Binarize the frame

    ret, binarized=cv2.threshold(M,90,255,cv2.THRESH_BINARY)    #am convertit la alb si negru

#Step 9 Get the coordinates of street markings on each side of the road!

    frame7=frame_o.copy()

    coloane=int(frame7.shape[1]*0.05)

    frame7[:,:coloane]=0
    frame7[:,frame7.shape[1]-coloane:frame7.shape[1]]=0
    first_half=frame7[:,:int(frame7.shape[1] / 2 )]
    last_half = frame7[:,int(frame7.shape[1] / 2) :]
    first_white=np.argwhere(first_half==225)   #le returneaza y x
    last_white = np.argwhere(last_half == 225)
    #print(len(last_half), (first_half))
    #print(frame7.shape[0]//2)

   #print (len(first_half), len(last_half))

    left_xs=[]
    left_ys=[]


    for x in first_white:
        left_xs.append(x[1])
        left_ys.append(x[0])

    right_xs=[]
    right_ys=[]

    for x in last_white:
        right_xs.append(x[1]+frame7.shape[0]//2)
        right_ys.append(x[0])

    left_xs=np.array(left_xs)
    left_ys = np.array(left_ys)
    right_xs = np.array(right_xs)
    right_ys = np.array(right_ys)

#Step10 Find the lines that detect the edge of the lane

    left_side=np.polynomial.polynomial.polyfit(left_xs,left_ys,deg=1)   # b and a from y=ax+b
    right_side=np.polynomial.polynomial.polyfit(right_xs,right_ys,deg=1)
    #print(left_side,right_side)

    def is_in (value, a , b, init_value=0):
        return (value-b)/a if -(10**8)<=(value - b)/a <= 10**8 else init_value

    left_top_y = 0
    left_top_x = is_in(left_top_y,left_side[1],left_side[0],left_top_x)

    left_bottom_y = frame7.shape[0]
    left_bottom_x = is_in(left_bottom_y,left_side[1],left_side[0],left_bottom_x)

    right_top_y =  0
    right_top_x = is_in(right_top_y,right_side[1],right_side[0],right_top_x)

    right_bottom_y = frame7.shape[0]
    right_bottom_x = is_in(right_bottom_y,right_side[1],right_side[0],right_bottom_x)

    left_top = int(left_top_y), int(left_top_x)
    left_bottom = int(left_bottom_y), int(left_bottom_x)
    right_top = int(right_top_y+205), int(right_top_x)
    right_bottom = int(right_bottom_y+205), int(right_bottom_x)

    #print(right_top, right_bottom)
    #print(frame7.shape)


    cv2.line(binarized,left_top,left_bottom,(200,0,0),5)
    cv2.line(binarized,right_top,right_bottom,(100,0,0),5)

    #cv2.line(binarized,(350,0),(350,280),(100,0,0),5)
    cv2.line(binarized,(240,0),(240,280),(255,0,0),1)

#Step11 Final Visualization

    frame8 = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    cv2.line(frame8, left_top, left_bottom, (255, 0, 0), 3)
    magic_matrix2 = cv2.getPerspectiveTransform(screen_bounds , trapezoid_bounds)
    frame8 = cv2.warpPerspective(frame8 , magic_matrix2 , (frame8.shape[1], frame8.shape[0]))
    white_left = np.argwhere(frame8 == 255)
    frame8_xs=[]
    frame8_ys=[]
    for x in white_left:
        frame8_xs.append(x[1])
        frame8_ys.append(x[0])


    frame9 = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    cv2.line(frame9, right_top, right_bottom, (255, 0, 0), 3)
    magic_matrix3 = cv2.getPerspectiveTransform(screen_bounds, trapezoid_bounds)
    frame9 = cv2.warpPerspective(frame9, magic_matrix3, (frame9.shape[1], frame9.shape[0]))
    white_right = np.argwhere(frame9 == 255)
    frame9_xs = []
    frame9_ys = []
    for x in white_right:
        frame9_xs.append(x[1])
        frame9_ys.append(x[0])

    #print(white_right)

    left_side_frame10 = np.polynomial.polynomial.polyfit(frame8_xs, frame8_ys, deg=1)
    right_side_frame10 = np.polynomial.polynomial.polyfit(frame9_xs, frame9_ys, deg=1)

    left_top_y_f = 0
    left_top_x_f = is_in(left_top_y_f, left_side_frame10[1], left_side_frame10[0], left_top_x_f)

    left_bottom_y_f = frame7.shape[0]
    left_bottom_x_f = is_in(left_bottom_y_f, left_side_frame10[1], left_side_frame10[0], left_bottom_x_f)

    right_top_y_f = 0
    right_top_x_f = is_in(right_top_y_f, right_side_frame10[1], right_side_frame10[0], right_top_x_f)

    right_bottom_y_f = frame7.shape[0]
    right_bottom_x_f = is_in(right_bottom_y_f, right_side_frame10[1], right_side_frame10[0], right_bottom_x_f)

    left_top_f = int(left_top_x_f), int(left_top_y_f)
    left_bottom_f = int(left_bottom_x_f), int(left_bottom_y_f)
    right_top_f = int(right_top_x_f ), int(right_top_y_f)
    right_bottom_f = int(right_bottom_x_f ), int(right_bottom_y_f)

    #print(left_top_f)

    frame10=frame_o.copy()
    cv2.line(frame10, left_top_f, left_bottom_f, (50, 50, 250), 2)
    cv2.line(frame10, right_top_f, right_bottom_f, (50, 250, 50), 2)












    # ret (bool): Return code of the `read` operation. Did we get an image or not?
    #             (if not maybe the camera is not detected/connected etc.)

    # frame (array): The actual frame as an array.
    #                Height x Width x 3 (3 colors, BGR) if color image.
    #                Height x Width if Grayscale
    #                Each element is 0-255.
    #                You can slice it, reassign elements to change pixels, etc.

    if ret is False:
        break

    cv2.imshow('Original', frame_o)
    cv2.imshow('Grey', frame)
    cv2.imshow('Trapezoid', frame2*255)
    cv2.imshow('Road', frame * frame2)  # suprapun cele 2 frame-uri
    cv2.imshow('Top-Down', frame3)
    cv2.imshow('Blur', frame4)
    cv2.imshow('Vertical', cv2.convertScaleAbs(frameV))
    cv2.imshow('Horizontal', cv2.convertScaleAbs(frameH))
    cv2.imshow('Sobel_Filter', M)
    cv2.imshow('binarized', binarized)
    cv2.imshow('CUt 5%', frame7)
    cv2.imshow('Final', frame10)




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
