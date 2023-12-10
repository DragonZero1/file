f=open('C:/Test/test.txt','r+')
choice_fuction=input('어떤 기능을 실행하시겠습니까?')
choice_line=int(input('몇번째 줄인지 선택해 주시기 바랍니다.'))

#파일 오픈을 할때 'W' 모드로 호출하게 되면 그 
# WRITE모드로 객체가 만들어진 순간 그 경로에 위치한 메모장의 내용이 싹 사라지고 나서 새로운 내용이 저장 될 수 있도록 준비 된다!!
#해당 파일의 내용이 모두 삭제되고 새로운 내용을 기록할 준비가 됩니다

#'r+' 모드를 사용하면 파일을 읽고 쓰기 모드로 열게 되므로, 
# 파일의 내용을 읽을 수 있으면서도 새로운 내용을 추가하거나 기존 내용을 변경할 수 있습니다.


def add(choice_line):
    contents=input('어떠한 내용을 집어 넣어시겠습니까?') 
    txt_list=f.read().split('\n')               #객체 f의 내용을 전체 다 문자열로 읽는데, 이때, split함수를 쓰게 되면 개행을 기점으로 하나씩 잘라서, 
                                                #개행을 기준을 자른 것들을 하나의 요소로서 각각 넣고 그것을 하나의 리스트로 만드는 함수!                        
                                                #file을 read를 했기 때문에 커서가 제일 마지막 까지 내려와 있다.
    txt_list.insert(choice_line-1,contents)
    f.seek(0)                           
    f.write("\n".join(txt_list)) # "특정 구분자".join(itrable 변수 or 객체) 함수는 문자열을 합치는 데 사용되며, 특정 구분자(separator)를 이용하여 여러 문자열을 결합합니다.
                                      #이 함수는 문자열로 구성된 리스트나 튜플 등의 모든 요소들 사이사이에 특정 구분자를끼워 넣고 그것을 하나의 문자열로 만든는 함수이다.
                                    
                                    #내용을 덮어 쓰고 싶다!! 이렇게 하면 그냥 밑에 내용을 추가 한거 밖에 안된다.
                                     #이때 방법이 존재하는게 f.seek(파일의 n번째 문자위치) #0부터시작해서, 개행,띄어쓰기 제외하고 하면된다.
                                    # 파일 커서의 위치를 이동시키는 것은 읽기 또는 쓰기 작업을 하는 위치를 변경하는 것이며, f.read() 또는 f.write() 등의 작업을 할 때 현재 커서 위치에서 시작됩니다.
                                    #덮어쓰고 나서 덧붙일 내용이 있을경우에 밑에 그냥 write /writeline사용해라!
    
def delete_line(choice_line):
    txt_list=f.read().split('\n')   #반드시 .read()를 써줘야 한다!! 
    txt_list[choice_line-1]=""
    f.seek(0)
    f.write("\n".join(txt_list))  #writelines을 쓰지 않고 write를 써야한다. join함수를 쓰게 되면 리스트가 하나의 문자열로 바뀌기때문에!
    
    
def copy_cover(choice_line):    
    cover_line=int(input("몇번째 줄에 덮어쓰실 건가요?"))-1
    txt_list=f.read().split('\n')
    temp=txt_list[choice_line-1]
    txt_list[cover_line]=temp
    f.seek(0)
    f.write("\n".join(txt_list))
    
    
if choice_fuction=='a':
    add(choice_line)
    
    
if choice_fuction=='b':
    delete_line(choice_line)
    
if choice_fuction=='c':
    copy_cover(choice_line)