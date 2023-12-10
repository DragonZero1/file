###문제 상황###
#:폴더 A에다가 랜덤이름으로 100개 만들고, 그 만들어진 100개의 폴더 안에 각각 폴더이름_01.txt라는 파일을 생성해라!


import names #이름 랜덤으로 생성 하기 위해서 > names.get.
import os # 폴더 만들기 위해서 >os.makedirs() (폴더 안에 폴더 만들어 준다.) // os.listdir('경로') >>경로 안에 있는 모든 것들을 파일이름 이랑 폴더이름랑 다 이것을 갖고 와서 list형태로 반환한다. 
                                                                            #+ 여기서 내가 .py로 된 파일만 뽑아내고 싶다면 또 다른 방법이 존재하는데!!
                                                                            #이때 사용하는 방법이 lambda 함수 + endswith('.py') >>문자열이 특정 접미사(뒷부분)로 끝나는지 여부를 확인하는 파이썬 문자열 메소드입니다. 
                                                                            #python_files = [file for file in all_files if file.endswith('.py')]
#파일을 만든다 > open함수를 쓰면된다!

for i in range(4): #range(0이상 4미만)=> 반복횟수는 총 4회
    name=names.get_full_name()
    os.makedirs(f'C:/Test/A/{name}')
    f=open(f"C:/Test/A/{name}/{name}_01.txt",'w') #open함수는 기본적으로 파일을 만드는 함수이다! 그래서 내가 만들고 싶은 파일의 확장자 명을 바꿔주면 이렇게 사용할 수 있다. 
    f=open(f"C:/Test/A/{name}/{name}_01.hwp",'w') #open함수는 기본적으로 파일을 만드는 함수이다! 그래서 내가 만들고 싶은 파일의 확장자 명을 바꿔주면 이렇게 사용할 수 있다.