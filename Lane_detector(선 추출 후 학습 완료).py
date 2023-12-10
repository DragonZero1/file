#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from sklearn.linear_model import RANSACRegressor

###트랙커 사용시 이렇게 쓰자###
# def onChange(x):
#     pass

# cv2.namedWindow('TrackBar Window')  # 윈도우 창 이름 설정
# cv2.createTrackbar('min', 'TrackBar Window', 0, 255, onChange)  # 트랙바 만들기
# cv2.createTrackbar('max', 'TrackBar Window', 0, 255, onChange)
# cv2.setTrackbarPos('min', 'TrackBar Window', 0)
# cv2.setTrackbarPos('max', 'TrackBar Window', 255)



# Get all images
pic_list=os.listdir('test_video/videoFrames/') #['1.jpg', '10.jpg', '100.jpg', '101.jpg', '102.jpg', '103.jpg',]
test_images = [mpimg.imread('test_video/videoFrames/' + i) for i in os.listdir('test_video/videoFrames/')] #이미지 객체가 순차적으로 리스트 형태로 저장이 된 상태
test_image_names = ['test_video/videoFrames/'+i for i in os.listdir('test_video/videoFrames/')] #그 이미지가 저장된 경로가 문자열의 리스트 형태로 저장이 된 상태

#GET IMAGE
im = test_images[0]  #test_imges 에는 이미지 객체가 리스트 형태로 저장이 된 상태이다. 따라서 이렇게 하나의 이미지만 뽑아 올 수 있다.
imshape = im.shape   #하나 가져온 이미지의 세로,가로, 채널수 를 imshape의 변수에 저장을 시킨다.


# -------------GREYSCALE IMAGE---------------
grayIm = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

#------------GAUSSIAN SMOOTHING----------------- #노이즈를 제거한다!
kernel_size = 9; # bigger kernel = more smoothing
smoothedIm = cv2.GaussianBlur(grayIm, (kernel_size, kernel_size), 0)

#-------------EDGE DETECTION---------------------
#cv2.Canny함수는 canny edge함수라고 불리며, 이것은 이미지의 egde 를 뽑을때 사용한다! 
#이렇게 Canny 함수를 써서 테두리 선을 뽑아낸다고 할때 가장 문제가 이게 한장의 사진으로 되어 있는데, 문제는 한장의 사진이 아니라 
#동영상이기 때문에 최적의 값 Minval , Maxval을 찾아내기가 힘들다! 이 해당 예제에서는, 60,150으로 두고있다.
#cv2.Canny(smoothedIm, minVal, maxVal)


minVal = 60
maxVal = 150
edgesIm = cv2.Canny(smoothedIm, minVal, maxVal)

'''
Canny 에지 검출은 이미지에서 에지를 찾는 기술로, 이때 강도가 임계값을 넘어가는 부분을 에지로 간주합니다.
minVal: 에지로 간주되는 최소 강도입니다. 이 값보다 작은 강도의 픽셀은 에지로 간주되지 않습니다.
maxVal: 강한 에지로 간주되는 최대 강도입니다. 이 값보다 큰 강도의 픽셀은 강한 에지로 간주됩니다.
minVal과 maxVal은 Canny 에지 검출 알고리즘에서 얼마나 세밀하고 강한 에지를 찾을지를 결정하는 매개변수로,
이 값들을 조절하여 원하는 에지 검출 결과를 얻을 수 있습니다. 너무 작은 값으로 설정하면 노이즈에 민감해질 수 있고,
너무 큰 값으로 설정하면 실제 에지를 놓칠 수 있습니다. 실험을 통해 적절한 값을 찾는 것이 일반적입니다.
위의 값의 범위는 0~255 까지 이다.
'''

#-------------------------CREATE MASK--------------------------------
'''
마스크를 만들어서 우리가 알고 싶은 값, roi(reasonable interest) 이것만 알고 싶은 영역 즉, 
차선이기 때문에 나머지 영역은 다 0으로 만들어 주면 roi를 더 쉽게 찾을 수 있다. 
따라서 마스크를 만들어서 그 마스크를 dege를 딴 그 이미지에다 씌우게 되면 우리가 보고싶은 영역인 roi를 잡아낼 수 있다. 
즉, 외부 적인 부분을 다 검정색으로 만들게 된다. 
즉, 마스크를 만들고 그 마스크를 이미지에 다가 씌우면 ROI만 출력된다. 
마스크를 만들고 그 마스크에다가 적용을 할때  cv2.bitwise_and 연산을 하게 되면 마스크를 씌운 부분을 제외하고는다 0이된다!  
'''
vertices = np.array([[(0,imshape[0]),(465, 320), (475, 320), (imshape[1],imshape[0])]], dtype=np.int32)  #. 만약 이미지 좌표를 다루는 경우에는 일반적으로 정수로 표현하는 것이 일반적이다.
'''
np.array가 지금 결국엔 넘파이를 이용해서 행렬을 만든것인데, 
넘파이가 행렬인데, 1차원 행렬은 보통 한점의 좌표를 나타내며, 
2차원 행렬의 경우 세로,가로가 있게 되며 한점이 아닌 여러점의 좌표를 표현 할 수 있다. 
따라서, 넘파이 배열의 경우, matplotlib에서는 2차원 넘파이 배열을 사용하여 여러 점을 한 번에 플로팅하거나 시각화할 수 있습니다.
예를 들어, 여러 점을 이어 선으로 연결하거나 다각형을 채우기 위해 사용됩니다.
dtype을 명시적으로 지정하지 않고 배열을 생성하면, numpy는 주어진 데이터를 기반으로 배열의 데이터 타입을 결정합니다. 
기본적으로 좌표의 값이 실수형일 경우, 배열의 데이터 타입은 부동 소수점이 됩니다.
''' 

mask = np.zeros_like(edgesIm)   
color = 255
cv2.fillPoly(mask, vertices, color)

'''
cv2.fillPoly(img, pts, color)  입력으로 받은 이미지 상에 주어진 점의 좌표를 이용하여 해당 다각형을 그린 뒤,
                                그 내부를 특정 색상으로 채우는 역할을 합니다.
img: 다각형이 채워질 이미지
pts: 만들 다각형의 좌표를 담은 배열
color:만든 다각형의 색을 지정
'''

#----------------------APPLY MASK TO IMAGE-------------------------------
maskedIm = cv2.bitwise_and(edgesIm, mask)
'''
즉, 마스크를 만들고 그 마스크를 이미지에 다가 씌우면 ROI만 출력된다. 
마스크를 만들고 그 마스크에다가 적용을 할때  cv2.bitwise_and 연산을 하게 되면 마스크를 씌운 부분을 제외하고는다 0이된다! 
마스크를 에 색이 들어가 있는공간은 1이고 그리고 edge에서 선을 추출한 부분만 1이기때문에 
둘다 1인것만 딱 출력이 되게 된다!  
'''

maskedIm3Channel = cv2.cvtColor(maskedIm, cv2.COLOR_GRAY2BGR)


# #-----------------------HOUGH LINES------------------------------------
'''
lines = cv2.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap)
image: 입력 이미지 (주로 에지 감지 등을 통해 얻은 이미지)
rho: Hough 공간에서의 거리 해상도
theta: Hough 공간에서의 각도 해상도 (라디안 단위)
threshold: 허프 변환에서 선으로 판단하기 위한 투표의 임계값
            Hough 변환은 이미지 상의 점들을 이용하여 직선을 찾는 알고리즘입니다. 
            알고리즘이 어떤 점들을 하나의 직선으로 간주할지 결정하는데, 이때 사용되는 기준이 threshold 값입니다.
minLineLength: 선으로 간주할 최소 길이
            HoughLinesP 함수는 이미지에서 직선을 찾는데 사용되며, 이때 얻어지는 선분들은 실제 직선이 아닌 끊어진 선분일 수 있습니다. 
            이렇게 감지된 모든 선분 중에서 minLineLength보다 길이가 짧은 선분은 제외됩니다.
maxLineGap: 하나의 선으로 간주할 최대 간격 =  하나의 선으로 간주할 수 있는 선분 간의 최대 거리(간격)
            선분이 이 최대 간격보다 작다면 해당 선분들은 하나의 선으로 간주되어 합쳐집니다.
            만약 간격이 max_line_gap보다 크다면, 이는 서로 다른 선분으로 처리됩니다. 
            즉, max_line_gap이 크면 선분들이 서로 떨어져 있어도 하나의 선으로 간주할 가능성이 높아지고, 
            작으면 더 가까이 있는 선분들도 하나의 선으로 처리됩니다.
            max_line_gap을 조절하면 감지된 선분들이 어떻게 합쳐지는지에 대한 영향을 제어할 수 있습니다.

이 함수는 선분의 끝점을 반환합니다. 
반환된 lines는 선의 끝점들의 배열입니다. 각 선분은 [x1, y1, x2, y2] 
#선분의 시작점과 선분의 끝점이 하나의 리스트형태로 
lines = np.array([
    [[x1_1, y1_1, x2_1, y2_1]],
    [[x1_2, y1_2, x2_2, y2_2]],
    ...,
    [[x1_n, y1_n, x2_n, y2_n]]
], dtype=np.int32)
여기서 n은 검출된 선분의 수입니다. 각각의 [x1, y1, x2, y2]는 하나의 선분을 나타냅니다.
lines 배열의 각 원소는 한 개의 선분을 나타냅니다. 각 선분은 두 점으로 정의되는데, 각각 시작점과 끝점입니다.
각 선분은 [x1, y1, x2, y2] 형태로 표현되며, 여기서 (x1, y1)은 선분의 시작점이고, (x2, y2)은 선분의 끝점입니다.
'''

rho = 2 
theta = np.pi/180 
threshold = 100     # 만나는 점의 기준,(직선으로 판단할 최소한의 점의 개수 ) 숫자가 작으면 많은 선이 검출되지만 정확도가 떨어지고, 숫자가 크면 정확도가 올라감.
min_line_len = 100 
max_line_gap = 2000  
lines = cv2.HoughLinesP(maskedIm, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

# print(lines)
# Check if we got more than 1 line
if lines is not None and len(lines) > 2: #선분이 3개 이상 나오게 되면 아래에 코드를 실행하라! 
    # Draw all lines onto image
    allLines = np.zeros_like(maskedIm)
    for i in range(len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            cv2.line(allLines,(x1,y1),(x2,y2),(255,255,0),2) # plot line
    
    # Plot all lines found              #이미지 출력 4단계는 하나의 고정된 과정이다.
    plt.figure(7)                       #plt.figure() or(1) 가능하고, 하나의 창을 만든다. plt.figure 와 plt.imshow는 하나의 한쌍이라고 생각해라! 그리고 밑에 타이틀 창 만들어 주고 show 해준다.
    plt.imshow(allLines,cmap='gray')    ##이때 imshow (___)괄호 안에 내가 출력하고자 하는 이미지 객체를 넣어줘야 한다. 그리고 plt.imshow는 기본적으로 컬로로 출력을 하기 때문에 반드시 cmap='gray'로 바꿔줘야 흑백으로 출력된다.
    plt.title('All Hough Lines Found')  #이것은 figure라는 창 안에 하나의 사진이 출력될 것이고 그 사진의 title 제목을 붙여주는 기능을 한다.
    plt.show()                          #이걸 붙여줘야 figure창을 띄우는 기능을 하게된다.
    
    
    
    # if cv2.waitKey()==ord('q'): #<< 이렇게 적게되면, waitkey() 함수는 기본적으로 사용자로 부터 값을 입력받아서 아스키코드 값을 반환하는 함수 이다! 함수의 기능을 반드시 잘 이해해야 한다.
#       plt.close()     #이렇게 하면 q를 입력하게 되면 위의 창이 꺼지게 된다!! 

########################################################################


'''
 [[[618 387 869 538]]

 [[695 442 848 538]]

 [[235 477 433 338]]

 [[200 515 434 339]]

 [[605 385 677 429]]

 [[186 512 366 391]]

 [[604 378 650 405]]

 [[564 360 703 437]]

 [[187 513 430 337]]

 [[614 389 846 539]]

 [[199 515 279 457]]

 [[609 386 659 419]]]
'''

x_1 = lines[0:12, 0:1, 0:1].reshape(12,1) #2차원
y_1= lines[0:12, 0:1, 1:2].reshape(12)    #1차원
x_2= lines[0:12, 0:1, 2:3].reshape(12,1)  #2차원
y_2= lines[0:12, 0:1, 3:4].reshape(12)    #1차원

#print(y_1.ndim) #넘파이 배열의 차원 수를 알고 싶을때 사용하는 변수! 

model = RANSACRegressor()
model_1=RANSACRegressor()
# 모델 훈련
model.fit(x_1,y_1)
model_1.fit(x_2,y_2)

# 모델의 예측 결과
y_1_predict = model.predict(x_1)
y_2_predict = model_1.predict(x_2)

# 원본 데이터와 모델의 예측 결과 시각화
plt.scatter(x_1, y_1, label='DATA POINT')
plt.scatter(x_2, y_2, label='DATA POINT_1')

plt.plot(x_1, y_1_predict, color='red', label='RANSAC')
plt.plot(x_2, y_2_predict, color='blue', label='RANSAC_1')

plt.title("RANSAC MODEL")
plt.legend()
plt.show()
