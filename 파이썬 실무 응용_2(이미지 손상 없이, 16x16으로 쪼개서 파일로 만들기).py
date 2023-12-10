import cv2


img_src=cv2.imread('C:/practice_01/natural.png',cv2.IMREAD_COLOR) #520*520 가로 세로인데, 이것을 16x16으로 퍼즐을 조각 처럼 잘라야 하는데 문제는 16으로 520이 나누어 떨어지지않는다.
                                                                    #근데 이때 퍼즐로 쪼갠다 하더라도 원본이미지가 훼손 되지 않아야 하는데, 그것을 해결하기 위해서 밑에 처럼 그 원본이미지를 액자에 넣어줘야한다.
                                                                    
height, width=img_src.shape[:2]

result1=height%16  # 8 #높이가 8이라는건 위 ,아래로 4씩 증가시켜서 액자를 만들고!
result2=width%16 # 8 # 너비가 8이라는건 좌,우로 4씩 증가시킨 액자를 만들면 
                     # 원본 이미지가 훼손되지  하나의 액자에 담긴 결과 물이 이렇게 아래와 같이 나오게 된다.
                     # 이렇게 만들게 되면 520x520을 16x16퍼즐로 자를 수 있게 되기 때문에! 즉, 딱 나누어 떨어지게 된다는 말이다.
wrap_image=cv2.copyMakeBorder(img_src,4,4,4,4,borderType=cv2.BORDER_WRAP)


#딥러닝을 데이터 하나하나가 다 중요하기 때문에 원본이미지를 훼손 시키거나 사이즈를 별도로 변경시키는 일 없이 원본이미지에 훼손을 주어선 안된다. 따라서, 내가 다뤄야 할 원본 이미지를 건드리지 않고,
# 이 원본 이미지를 담을 수 있는 액자를 만들어서 하나의 액자에다가 그 원본이미지(변경시키지 않은!)를 그 액자 안에 넣는 것이 바로 이 padding의 목적이다!
# 주변에 추가되는 빈 영역이나 값을 채워 주는 것을 바로 패딩(padding)이라고 한다.

height_1,width_1=wrap_image.shape[:2]  #가로:528 , 세로:528

for j in range(16):
    for i in range(16):
        
        
        divided_img=wrap_image[i*33:(i+1)*33,j*33:(j+1)*33] 

        if j<8:
            cv2.imwrite(f'C:/practice_01/0/divided_img_{i}_{j}.png',divided_img)
        else:
            cv2.imwrite(f'C:/practice_01/1/divided_img_{i}_{j}.png',divided_img)
            
            
    
     #f=open(f'C:/practice_01/0/{divided_img}.png','+r') #open함수를 써서 파일을 만들 수 는 있지만, open함숭의 경우 보통, 텍스트 파일을 다룰때 많이 사용한다. hwp txt 등의 파일을 많이 쓰기 때문에!
     #그것을 해결하기 위해서, cv2.imwrite함수를 쓴다. 이 함수는 opencv라이브러리 에서 이미지를 파일로 저장하는데 특화된 함수이기 때문에 이건 외워도라! 
     #cv2.imwrite(filename, img)
     #filename: 저장할 파일의 경로와 파일 이름을 지정합니다.
     #img: 저장할 이미지 데이터 
     
     #이름 사이에는 _바를 넣는다. 공백 대신에 넣는다.
     #i ,j , k식으로 변수 이름을 사용한다.