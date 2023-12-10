###문제 상황###
#:파일 .py 파이썬 파일을 100개를 만들어서 각 10개씩 끊어서 총 10개의 폴더에다가 담아라! 이때 폴더 이름은 랜덤으로 만들어져 있어야 한다.

import names
import os
import shutil #shutil.move('원본 경로','대상 경로') #원본 경로에 있는 파이썬 파일을 대상 경로에 로 이동시켜라!

for i in range(10):
    for i in range(10):
        f=open(f'C:/Test/{i}.py','w')
    f.close() #위에서 open을 써서 객체를 만들어 줬으면 반드시 그 객체를 다쓰고 나서 닫아줘야 한다!! 이건 중요!!
            #이렇게 닫아줘야 경로상의 충돌이 일으키지 않는다.
    file_list=os.listdir('C:/Test') #['0.py', '1.py', '2.py', '3.py', '4.py', '5.py', '6.py', '7.py', '8.py', '9.py']
    name=names.get_full_name()
    os.makedirs(f'C:/{name}')
    for files in file_list:
        shutil.move('C:/Test'+'/'+files,f'C:/{name}'+'/'+files)
        #문자열 끼리 더해지는것을 이용한것이다!
        #함수를 사용할 때는 반드시 이동시키려는 파일의 정확한 경로와 이름을 지정해야 합니다.
        # 이 함수는 첫 번째 매개변수로 이동시킬 파일이나 디렉토리의 현재 경로와 이름을,
        # 두 번째 매개변수로는 이동하고자 하는 대상 경로와 이름을 받습니다.

