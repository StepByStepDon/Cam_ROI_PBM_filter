#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 카메라로 관찰하고자 하는 대상을 30p만 녹화한다.
import cv2
import numpy as np

cap = cv2.VideoCapture(0) # camera 녹화시

# 카메라 width, height
w = int(cap.get(3))
h = int(cap.get(4))
size = (w,h)
fps = cap.get(cv2.CAP_PROP_FPS)


# videowriter 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
file_name = './input_1.avi' # ROI 지정할 것
out = cv2.VideoWriter(file_name , fourcc, fps, size)

frame_counter = 1
# 30p 만 읽기
if cap.isOpened():
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, -1)
        # 가우시안 필터 & 노이즈 제거
        # blur1 = cv2.GaussianBlur(img, (3,3), 0)
        
        # bilateral filter & 노이즈 제거 & 선명함
        blur2 = cv2.bilateralFilter(img, 5, 75, 75)
        
#         emerged = np.hstack((img, blur2))
        
        if ret:
#             print("Current frame is {}".format(frame_counter))
            cv2.imshow('Bilateral', blur2)
            out.write(blur2)
            
            frame_counter += 1
        
        # 30p 받아서 저장하기 위함
        if frame_counter == 150 :
            out.write(blur2)
            break
        
        
        if cv2.waitKey(66) & 0xFF == 27:
            break
            
else:
    print('no camera')
    
cap.release()
out.release()
cv2.destroyAllWindows()


# In[57]:


# 녹화한 거 읽어오기 & ROI 지정해서 사이즈 줄여서 다시 저장
# "file_name" 가져오기
import cv2
import numpy as np

# 저장한 video 가져오기
cap = cv2.VideoCapture(file_name)
fps = cap.get(cv2.CAP_PROP_FPS)

# 첫 프레임 읽어서, ROI 지정
win_name = 'ROI'
if cap.isOpened():
    ret, img = cap.read()
    
    # ROI 지정
    x, y, w, h = cv2.selectROI(win_name, img, False)
    
    
    # videowriter_roi 객체 생성
    size = (int(w), int(h))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    file_name = './input_1_roi.avi' 
    out = cv2.VideoWriter(file_name , fourcc, fps, size)

    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1) # 첫프레임에서 ROI 지정하고, 다시 시작
    while True:
        ret, img = cap.read()
        
        roi = img[y:y+h, x:x+w]
        cv2.imshow('Test', roi)
        out.write(roi)
        
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cv2.imshow('Test', roi)
            out.write(roi)
            break
        
        if cv2.waitKey(66) & 0xFF == 27:
            break
else:
    print('There\'s no Video')
cap.release()
out.release()
cv2.destroyAllWindows()


# In[58]:


# Importing modules
import cv2
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from numpy.fft import fft, ifft
from numpy import tile, real, min, zeros
import numpy as np

from scipy.signal import firwin, convolve
from skimage.filters import gaussian

import math

M_PI = math.pi
eps = 2**(-52)


# In[ ]:


# ROI 된 Video PBM 실행하기
fileName = file_name

# PBM parameter setting - alpha, Low_Freq, High_Freq
Low_Freq = float(input('Low Frequency ='))
High_Freq = float(input('High Frequency = '))
alpha = float(input('alpha = '))
Sampling_Rate = int(input('Sampling Rate = '))
# Sampling_Rate = 2200
refFrame = 0
NumberOfPyramid = 2
Orientation = 2
sigma = 5
attenuateOtherFreq = 0
ratioOfFrame = 1


# Video Information Checking
cap_original = cv2.VideoCapture(fileName)

ret, frame = cap_original.read()
frameX = frame.shape[1]
frameY = frame.shape[0]
nColor = frame.shape[2]
nFrame = int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT))

print(nFrame)
print(frameX)
print(frameY)

# ready_1
pyrLayers =NumberOfPyramid
rVals =[1]
for i in range(pyrLayers):
    rVals.append(0.5**(i+1))


# In[ ]:


import cv2
import numpy as np
# ready_2 - Definition
def getFilters(dimension, rVals, orientations):
    X = int(dimension[1])
    Y = int(dimension[0])

    defaultTwidth = 1
    twidth = defaultTwidth

    polarGrid = getPolarGrid(dimension)
    count = 0

    angle = np.array(polarGrid[0]) #픽셀 위치별 각도 값
    rad = np.array(polarGrid[1]) #픽셀 위치별 거리 값

    mask = getRadialMaskPair(rVals[0], rad, twidth)
    himask = mask[0]
    lomaskPrev = mask[1]

    filters = []
    filters.append(himask)

    for k in range(1, len(rVals)):
        mask = getRadialMaskPair(rVals[k], rad, twidth)
        himask = mask[0]
        lomask = mask[1]

        radMask = himask*lomaskPrev

        for j in range(1, orientations+1):
            angleMask = getAngleMask(j, orientations, angle)
            filters.append(radMask*angleMask/2)

        lomaskPrev = lomask
    filters.append(lomask)

    for k in range(len(filters)):
        filters[k] = np.array(filters[k])
    return filters

#화면의 너비x높이 크기의 극좌표르를 2차원 배열으로 형성한다.
def getPolarGrid(dimension):
    X = dimension[1]
    Y = dimension[0]
    centerX = int(X / 2)
    centerY = int(Y / 2)

    # Create rectangular grid
    xramp = np.array([ [(x-int(X/2))/(X/2) for x in range(X)] for y in range(Y)])
    yramp = np.array([ [(y-int(Y/2))/(Y/2) for x in range(X)] for y in range(Y)])
    angle = np.arctan2(xramp, yramp)+M_PI/2

    rad = np.sqrt(xramp**2+yramp**2)
    rad[centerY][centerX] = rad[centerY-1][centerX]

    polarGrid = [angle, rad]
    return polarGrid

#3차원 그래프상으로 보이는 완전 꼬깔 형태에서, 위 아래의 범위를 잘라내고 원뿔대 같은 형태도 만든다.
def getRadialMaskPair(r, rad, twidth): 
    X = int(rad.shape[1])
    Y = int(rad.shape[0])

    log_rad = np.log2(rad)-np.log2(r)

    himask = log_rad
    himask[himask>0] = 0
    himask[himask<-twidth] = -twidth
    himask = himask*M_PI/(2*twidth)

    himask = np.cos(himask)
    lomask = np.sqrt(1-himask**2)

    mask = [himask, lomask]
    return mask

def getAngleMask(b, orientations, angle):
    order = orientations - 1
    const = (2 ** (2 * order)) * (math.factorial(order) ** 2) / (orientations * math.factorial(2 * order)) # Scaling constant

    angle_ = (M_PI + angle - (M_PI * (b - 1) / orientations)) % (2 * M_PI) - M_PI
    anglemask = 2 * np.sqrt(const) * (np.cos(angle_) ** order) * (abs(angle_) < (M_PI / 2))  # Make falloff smooth
    return anglemask

filters = getFilters([frameY, frameX], rVals, Orientation)

def getFilterIDX2(filters, orientations, rVals):
    X = filters[0].shape[1]
    Y = filters[0].shape[0]
    nFilts = len(filters)
    filtIDX = [[None for j in range(orientations)] for i in range(nFilts)]
    croppedFilters = []

    #himask IDX
    filtIDX[0][0] = [y for y in range(Y)]
    filtIDX[0][1] = [x for x in range(X)]
    croppedFilters.append(filters[0])

    #stearable filter IDX
    for k in range(1, nFilts-1, orientations):
        n = int(k/2)+1
        lb_y = int( (Y*(np.sum(rVals[0:n])-1))/2 )
        ub_y = Y - lb_y
        lb_x = int( (X*(np.sum(rVals[0:n])-1))/2 )
        ub_x = X - lb_x

        for i in range(orientations):
            filtIDX[k+i][0] = [y + lb_y for y in range(ub_y - lb_y)]
            filtIDX[k+i][1] = [x + lb_x for x in range(ub_x - lb_x)]

        for i in range(orientations):
            croppedFilters.append(filters[k+i][lb_y:ub_y, lb_x:ub_x])


    #lomaskIDX
    lb_y = int( (Y * (np.sum(rVals) - 1))/2 )
    ub_y = Y - lb_y
    lb_x = int( (X * (np.sum(rVals) - 1))/2 )
    ub_x = X - lb_x


    filtIDX[nFilts - 1][0] = [y + lb_y for y in range(ub_y - lb_y)]
    filtIDX[nFilts - 1][1] = [x + lb_x for x in range(ub_x - lb_x)]
    croppedFilters.append(filters[nFilts-1][lb_y:ub_y, lb_x:ub_x])

    filterIDX = [croppedFilters, filtIDX]
    return filterIDX


filterIDX = getFilterIDX2(filters, Orientation, rVals)
croppedFilters = filterIDX[0]
filtIDX = filterIDX[1]

def bgr2yiq(img):
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    I = 0.596 * R - 0.274 * G - 0.322 * B
    Q = 0.211 * R - 0.523 * G + 0.312 * B

    img_yiq = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
    img_yiq[:, :, 0] = Y
    img_yiq[:, :, 1] = I
    img_yiq[:, :, 2] = Q
    #YIQ = np.array([[0.299, 0.587, 0.114],[0.596, -0.274, -0.322],[0.211, -0.523, 0.312]])
    return img_yiq

def yiq2bgr(img):
    Y = img[:, :, 0]
    I = img[:, :, 1]
    Q = img[:, :, 2]

    R = Y+0.956*I+0.621*Q
    G = Y-0.272*I-0.647*Q
    B = Y-1.106*I+1.703*Q

    img_rgb = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
    img_rgb[:, :, 0] = B
    img_rgb[:, :, 1] = G
    img_rgb[:, :, 2] = R
    #YIQ = np.array([[0.299, 0.587, 0.114],[0.596, -0.274, -0.322],[0.211, -0.523, 0.312]])
    return img_rgb

vidFFT = np.zeros([nFrame,frameY,frameX], dtype=np.complex64)

for k in range(0, nFrame):
    clipMat = frame/255
    tVid = bgr2yiq(clipMat)[:,:,0]
    vidFFT[k] = fftshift(fft2(tVid))
    ret, frame = cap_original.read()
##################################################################  
def iir_window_bp_2(delta, fl, fh):
    length = delta.shape[0] + 1
    b = firwin(length, (fl * 2, fh * 2), pass_zero=False)[0:length - 1]

    # temp = fft(ifftshift(b))
    # out = pixelConvSame(delta, ifftshift(b))
    out = np.apply_along_axis(lambda m: np.convolve(m, ifftshift(b), mode='same'), axis=0, arr=delta)
    return out
    
    
####################################################################    


def amplitude_weighted_blur(x, weight, sigma):
    if sigma != 0:
        return gaussian(x*weight, sigma, mode="wrap") / gaussian(weight, sigma, mode="wrap")
    return x

magnifiedLumaFFT = np.zeros([nFrame, frameY, frameX], dtype=np.complex64)
numLevels = len(filters)

for level in range(1, numLevels-1):
    X = len(croppedFilters[level][0])
    Y = len(croppedFilters[level])
    lb_x = filtIDX[level][1][0]
    ub_x = filtIDX[level][1][-1]+1
    lb_y = filtIDX[level][0][0]
    ub_y = filtIDX[level][0][-1]+1

    # 1. 기준프레임 설정
    clipMat = croppedFilters[level] * vidFFT[refFrame][lb_y:ub_y, lb_x:ub_x]
    pyrRef = ifft2(ifftshift(clipMat))
    pyrRefPhaseOrig = pyrRef / abs(pyrRef)
    pyrRef = np.angle(pyrRef)

    delta = np.zeros([nFrame,Y,X], dtype=np.float16)
    matCheck = []
    
    # 2. 각 프레임간 차이 계산
    for frameIDX in range(0, nFrame):
        filterResponse = ifft2(ifftshift( croppedFilters[level] * vidFFT[frameIDX][lb_y:ub_y, lb_x:ub_x] ))
        pyrCurrent = np.angle(filterResponse)
        clipMat1 = pyrCurrent - pyrRef
        clipMat2 = M_PI + clipMat1
        clipMat3 = clipMat2%(2*M_PI)
        clipMat4 = clipMat3 - M_PI
        clipMat = clipMat4
        delta[frameIDX] = clipMat

    # 3. 픽셀 변화 양상에 대한 band pass filtering
    delta_1 = iir_window_bp_2(delta, Low_Freq / Sampling_Rate, High_Freq / Sampling_Rate)  # Finite Impulse Response filter

    for frameIDX in range(0, nFrame):
        Phase = delta_1[frameIDX]

        originalLevel = ifft2(ifftshift(croppedFilters[level]*vidFFT[frameIDX][lb_y:ub_y, lb_x:ub_x]))

        if (sigma != 0):
            Phase = amplitude_weighted_blur(Phase, abs(originalLevel)+eps, sigma)
        
        # 4. alpha 계수 곱 및 병합
        Phase = alpha*Phase

        if (attenuateOtherFreq):
            tempOrig = abs(originalLevel)*pyrRefPhaseOrig
        else:
            tempOrig = originalLevel

        tempTransformOut = np.exp(1j*Phase)*tempOrig


        A = croppedFilters[level]
        B = fftshift(fft2(tempTransformOut))
        curLevelFrame = 2 * A * B

        matClip = magnifiedLumaFFT[frameIDX][lb_y:ub_y, lb_x:ub_x]
        magnifiedLumaFFT[frameIDX][lb_y:ub_y, lb_x:ub_x] = matClip + curLevelFrame

#5. lowpass residual 처리
level = len(filters)-1
lb_x = filtIDX[level][1][0]
ub_x = filtIDX[level][1][-1] + 1
lb_y = filtIDX[level][0][0]
ub_y = filtIDX[level][0][-1] + 1
for frameIDX in range(0, nFrame):
    lowpassFrame = vidFFT[frameIDX][lb_y:ub_y, lb_x:ub_x]*(croppedFilters[level]**2)
    matClip = magnifiedLumaFFT[frameIDX][lb_y:ub_y, lb_x:ub_x] + lowpassFrame
    magnifiedLumaFFT[frameIDX][lb_y:ub_y, lb_x:ub_x] = matClip


# In[ ]:


cap_original = cv2.VideoCapture(fileName)

fps = cap_original.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
FileName = './roi_1.avi'
obj_out = cv2.VideoWriter(FileName, fourcc, fps, (frameX, frameY))

ret, frame = cap_original.read()

for k in range(0, nFrame):
    clipMat1 = magnifiedLumaFFT[k]
    clipMat = ifft2(ifftshift(clipMat1))
    magnifiedLuma = np.real(clipMat)
    #명암에 대한 데이터만 적용하기 위해 yiq변환해서 Y 값만 가져온다.
    clipMat2 = bgr2yiq(frame)
    clipMat3 = magnifiedLuma*255
    clipMat2[:,:,0] = clipMat3
    
    #그 외의 Q, I는 그대로 사용할 것이므로 변환된 Y만 원본 영상에 대입한 뒤 다시 RGB로 변환한다.
    outFrame = yiq2bgr(clipMat2)
    
    outFrame[outFrame > 255] = 255
    outFrame[outFrame < 0] = 0
    outFrame = np.uint8(outFrame)
    obj_out.write(outFrame)
    ret, frame = cap_original.read()

cap_original.release()
obj_out.release()


# In[ ]:


# 동시 display
cap1 = cv2.VideoCapture(fileName)
cap2 = cv2.VideoCapture(FileName)

w = int(cap1.get(3))
h = int(cap1.get(4))
black = np.zeros((h, w, 3), dtype = np.uint8)
FileName2 = './roi_2.avi'

out = cv2.VideoWriter(FileName2, fourcc, fps, (2*w, 3*h))

while True:
    ret, frame1 = cap1.read()
    ret, frame2 = cap2.read()
    
    # original - PBM
    sum1 = np.hstack((frame1, frame2))
    
#     # 평균블러링
#     blur_av1 = cv2.blur(frame1, (10,10))
#     blur_av2 = cv2.blur(frame2, (10,10))
#     sum2 = np.hstack((blur_av1, blur_av2))
    
    # 가우시안블러링
    blur_gau1 = cv2.GaussianBlur(frame1, (3, 3), 0)
    blur_gau2 = cv2.GaussianBlur(frame2, (3, 3), 0)
    sum3 = np.hstack((black, blur_gau2))
    
#     # 미디언블러링
#     blur_median1 = cv2.medianBlur(frame1, 5)
#     blur_median2 = cv2.medianBlur(frame2, 5)
#     sum4 = np.hstack((blur_median1, blur_median2))
    
    # 바이레터럴 블러링
    blur_bi1 = cv2.bilateralFilter(frame1, 5, 75, 75)
    blur_bi2 = cv2.bilateralFilter(frame2, 5, 75, 75)
    sum5 = np.hstack((black, blur_bi2))
    
    sum = np.vstack((sum1, sum3, sum5))
#     sum = cv2.pyrUp(sum)
#     sum = cv2.pyrDown(sum)
    cv2.putText(sum, 'Original', (0,h-5), 1, 1, (255,255,255), 2)
    cv2.putText(sum, 'PBM', (w,h-5), 1, 1, (255,255,255), 2)
    
#     cv2.putText(sum, 'Original with Blur', (0,2*h-5), 1, 1, 5)
#     cv2.putText(sum, 'PBM with Blur', (w,2*h-5), 1, 1, 5)
    
#     cv2.putText(sum, 'Original with Gaussian Blur', (0,2*h-5), 1, 1, (255,255,255), 2)
    cv2.putText(sum, 'PBM with Gaussian Blur', (w,2*h-5), 1, 1, (255,255,255), 2)
    
#     cv2.putText(sum, 'Original with Median Blur', (0,4*h-5), 1, 1, 5)
#     cv2.putText(sum, 'PBM with Median Blur', (w,4*h-5), 1, 1, 5)
    
#     cv2.putText(sum, 'Original with Bilateral Blur', (0,3*h-5), 1, 1, (255,255,255), 2)
    cv2.putText(sum, 'PBM with Bilateral Blur', (w,3*h-5), 1, 1, (255,255,255), 2)
    
    cv2.imshow('Sum', sum)
    out.write(sum)
    
    if cap1.get(cv2.CAP_PROP_POS_FRAMES) == cap1.get(cv2.CAP_PROP_FRAME_COUNT):
        cv2.imshow('Sum', sum)
        out.write(sum)
        cap1.set(cv2.CAP_PROP_POS_FRAMES, 1)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 1)
    
    if cv2.waitKey(100) & 0xFF == 27:
        break
cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




