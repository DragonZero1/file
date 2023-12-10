import cv2
import numpy as np

img_src=cv2.imread('C:/practice_01/natural.png',cv2.IMREAD_COLOR)                                     
height, width=img_src.shape[:2]
result1=height%16
result2=width%16
wrap_image=cv2.copyMakeBorder(img_src,4,4,4,4,borderType=cv2.BORDER_WRAP)
height_1,width_1=wrap_image.shape[:2]  #가로:528 , 세로:528


black_background=np.zeros((height_1,width_1,3),dtype=np.uint8) #3차원 검정색 화면을 만들기 위해서 이렇게 하였다.

for j in range(16):
    for i in range(16):
        if j < 8:
            divided_img = cv2.imread(f'C:/practice_01/0/divided_img_{i}_{j}.png', cv2.IMREAD_COLOR)  
            black_background[i * 33:(i + 1) * 33, j * 33:(j + 1) * 33, :] = divided_img #반드시 3차원으로 인덱싱 해줘야 한다. 왜냐하면 원본이 3채널 bgr이기 때문에 
        else:
            divided_img = cv2.imread(f'C:/practice_01/1/divided_img_{i}_{j}.png', cv2.IMREAD_COLOR)
            black_background[i * 33:(i + 1) * 33, j * 33:(j + 1) * 33, :] = divided_img


#크롭 (Crop): 이미지 처리에서 주로 사용되며, 원본 이미지에서 일부 영역을 선택하는 작업을 나타냅니다. 
# 크롭된 이미지는 원본 이미지의 일부분이며, 원하는 영역을 추출하여 새로운 이미지로 만드는 것을 의미합니다. #슬라이싱과 유사하나 이것은 넘파이 배열 즉 배열을 슬라이싱 할때 많이 사용된다.


img_crop=black_background[3:width_1-4,3:height_1-4] #8만큼 커졌기 때문에, 사실 위,아래, 왼쪽,오른쪽 다 4씩 증가한 형태이다. 따라서 커진 이미지에서 원본 이미지만 뽑아내려면 시작점이 인덱스값으로 4부터시작 하기때문에
                                                        #3, widthe_1-4를 해준것이고 같은 원리로 높이도 3, height_1-4를 해주면 
cv2.imshow('result', black_background)
cv2.imshow('crop_img',img_crop) 
cv2.waitKey(0)
cv2.destroyAllWindows()