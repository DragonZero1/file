import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RANSACRegressor, LinearRegression

# LinearRegression은 통계학과 기계 학습에서 사용되는 회귀 분석
#(변수 x에 따른 종속 변수 y 의 출력값을 나타내는 하나의 학습모델) 의 한 유형입니다.
# 회귀 분석은 독립 변수(X)와 종속 변수(Y) 간의 관계를 모델링하고 예측하는 데 사용됩니다.
# LinearRegression은 특히 선형 관계를 모델링하는 데 사용됩니다.

'''
선형 관계는 두 변수 간의 관계가 직선 형태로 표현되는 것을 의미합니다. 따라서 한 변수가 다른 변수에 대해 일정한 비율로 변할 때 선형 관계가 있다고 말할 수 있습니다.
'''




# 데이터 생성
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# # 산점도 그리기
# result=plt.scatter(x, y)

# # 그래프 제목과 축 레이블 추가
# plt.figure(1)
# plt.title('Scatter Plot Example')
# # plt.xlabel('X-axis Label')
# # plt.ylabel('Y-axis Label')

# # 그래프 보여주기
# plt.show()



########################################
# 예제 데이터 생성
# X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # 입력 변수(-1,1) : -1은 넘파이 배열 1차원 배열을 자동적으로 배열을 바꿔라 // 1의 의미는: 각 행에 하나의 열을 가지도록 하라는 의미입니다.
# #                                                               즉, 배열의 각 원소가 하나의 열로 들어가도록 2차원 배열로 변환합니다. 
# '''
# [[1],
#  [2],
#  [3],
#  [4],
#  [5]]
# '''

# y = np.array([2, 3, 5, 7, 11])  # 출력 변수

# # 선형 회귀 모델 생성
# model = LinearRegression()

# # 모델 훈련
# model.fit(X, y)

# # 훈련된 모델을 사용하여 예측
# y_pred = model.predict(X)

# # 회귀 계수와 절편 출력
# print("회귀 계수 (기울기):", model.coef_)
# print("절편:", model.intercept_)
# y_pred = model.predict(X)   #[ 1.2  3.4  5.6  7.8 10. ]
# print(y_pred)

# mse = mean_squared_error(y, y_pred) #D오차 계산!
# print("Mean Squared Error:", mse)


# # 산점도로 실제 데이터 표시
# plt.scatter(X, y, color='blue', label='real')

# # 선형 회귀 모델이 학습한 선을 그래프에 추가
# plt.plot(X, y_pred, color='red', linewidth=2, label='estimate')

# # 그래프에 레이블과 범례 추가
# plt.xlabel('input (X)')
# plt.ylabel('output (y)')
# plt.legend()

# # 그래프 표시
# plt.show()


'''
좋은 질문입니다. Scikit-learn의 LinearRegression 모델은 입력 데이터 X를 2차원 배열로 기대합니다. 따라서 X를 2차원으로 변환해야 합니다. 이는 reshape(-1, 1)을 사용하여 간단하게 수행됩니다.

선형 회귀 모델을 사용할 때 왜 2차원 배열이 필요한지에 대해 이해하려면 모델의 입력 데이터에 대한 기대 형태를 고려해야 합니다. 선형 회귀 모델에서 입력 변수 X는 다음과 같은 형태를 가정합니다:

X는 2차원 배열이어야 합니다.
각 행은 하나의 데이터 포인트를 나타내며, 각 열은 해당 데이터 포인트의 특성을 나타냅니다.
따라서 reshape(-1, 1)을 사용하여 1차원 배열인 X를 2차원 배열로 변환하는 것은 모델의 입력 형태와 일치하게 만들기 위한 것입니다.

만약 하나의 특성(변수)만 있는 경우에도 반드시 2차원 배열을 사용해야 하는 것은 아니지만, Scikit-learn의 관례에 따라서 특성이 하나인 경우에도 2차원 배열로 입력하는 것이 권장됩니다. 
이는 코드를 일관성 있게 유지하고 여러 모델과 라이브러리를 사용할 때 편의성을 높이기 위한 것입니다.
'''

'''
그럼 다음 질문이 있는데, 선형회귀 모델이 결국엔 X값을 집어 넣게 되면 결국 y 값을 출력하게 되는데,model.fit(X, y)이렇게 둘다 집어넣는거야? 
###############################################################################################################################
ChatGPT
올바른 관찰입니다. model.fit(X, y)에서 X와 y를 함께 전달하는 것은 선형 회귀 모델을 학습시키기 위한 것입니다.

여기에 대한 이유는 다음과 같습니다:

X와 y의 관계 학습: 선형 회귀 모델은 입력 변수 X와 출력 변수 y 간의 관계를 학습해야 합니다. 
<<<fit 메서드를 호출할 때, 모델은 주어진 입력 변수 X와 해당하는 출력 변수 y 사이의 관계를 학습합니다.>>>
>>>그래서 출력값이 그 선의 기울기와 절편 즉, y=ax+b 할때 b의 값이라는거다.
모델 파라미터 추정: 학습 중에 모델은 기울기(coefficient)와 y축 절편(intercept)과 같은 모델의 파라미터를 조정하여 데이터에 가장 잘 맞는 선을 찾습니다.
이렇게 찾은 선은 주어진 X 값에 대해 적절한 y 값을 예측하는 데 사용됩니다.
'''

'''
그럼 이게 잘만들어진건지 아닌지는 어떻게판단해? y = np.array([2, 3, 5, 7, 11]) 이 값이랑 비교해서 값이 많이 차이가 안나면 학습이 잘된거야?

ChatGPT
맞습니다. 모델이 얼마나 잘 학습되었는지를 판단하기 위해서는 학습된 모델의 예측값과 실제 값 간의 차이를 살펴보는 것이 중요합니다. 이 차이는 일반적으로 "잔차" 또는 "오차"라고 불립니다. 
잔차는 실제 값과 모델이 예측한 값 간의 차이를 나타내며, 작을수록 모델이 더 정확하게 예측하고 있다고 볼 수 있습니다.

잔차를 계산하는 방법 중 하나는 각 예측값에서 실제 값 빼기입니다. 예를 들어, 다음과 같이 수행할 수 있습니다:

python
Copy code
residuals = y - y_pred
print(residuals)
그리고 이 잔차 값들을 통해 모델의 성능을 평가할 수 있습니다. 일반적으로는 잔차의 제곱합이나 평균 제곱 오차(Mean Squared Error, MSE)와 같은 지표를 사용합니다.
이 값이 작을수록 모델의 예측이 실제 값에 가깝다고 할 수 있습니다.


#################################################################################################################################
Scikit-learn에서는 mean_squared_error 함수를 통해 MSE를 계산할 수 있습니다. 이를 통해 모델의 성능을 수치화하고 비교할 수 있습니다:

python
Copy code
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)
MSE 값이 작을수록 모델이 잘 학습되었다고 판단할 수 있습니다.
#################################################################################################################################
'''



















'''
이상치에 강함: RANSAC은 데이터에서 이상치(Outlier)의 영향을 최소화하는 데 중점을 둡니다. 모델을 학습시킬 때 일부 데이터 포인트를 무시하면서 이상치에 영향을 받지 않도록 설계되었습니다.

랜덤 샘플링: RANSAC은 초기에 무작위로 샘플을 선택하여 모델을 구축하고, 그 후에 이 모델이 데이터에 얼마나 잘 맞는지를 평가합니다. 이를 반복하여 가장 잘 맞는 모델을 찾습니다.

자동적인 모델 선택: RANSAC은 반복적인 과정에서 가장 잘 맞는 모델을 자동으로 선택합니다. 따라서 사용자가 모델의 파라미터를 설정할 필요가 없습니다.
'''

# ################################################
# # 선형 회귀 모델 생성
# model = RANSACRegressor(base_estimator=LinearRegression(), random_state=42)
# '''
# 예, 정확하게 말하면 맞습니다. RANSACRegressor의 base_estimator 매개변수는 RANSAC이 내부에서 사용하는 기본 회귀 모델을 정의합니다. 
# 여기서는 선형 회귀 모델(LinearRegression())을 base_estimator로 사용하고 있습니다.
# RANSAC은 무작위로 데이터의 일부를 선택하고, 그 선택된 일부를 사용하여 base_estimator로 지정한 모델을 훈련합니다.
# 그 후에는 이 모델이 전체 데이터에 얼마나 잘 맞는지를 평가하고, 잘 맞는 데이터 포인트들을 모델에 포함시킵니다. 이 과정을 지정한 반복 횟수나 정지 조건에 도달할 때까지 반복합니다.
# '''

'''
ANSACRegressor는 내부적으로 기본적으로 선형 회귀 모델을 사용합니다. 따라서 별도의 base_estimator를 지정하지 않고도 RANSACRegressor를 사용할 수 있습니다.
'''

# 데이터 생성 ##단순하게 x,y값 을 만든 자료형에 불과함!
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 2 * x + 1 + np.random.normal(0, 2, size=len(x))


# 선형 회귀 모델 생성
model = RANSACRegressor(random_state=42)

# 모델 훈련
x_reshape = x.reshape(-1, 1)
model.fit(x_reshape, y)

# 모델의 예측 결과
y_pred = model.predict(x_reshape)

# 원본 데이터와 모델의 예측 결과 시각화
plt.scatter(x, y, label='DATA POINT')
'''
산점도는 차트나 그래프 상에 변수 x와 y의 값을 가진 각 데이 터 포인트를 점으로 나타냅니다.
x와 y는 각각 x 및 y 좌표를 포함하는 배열이나 리스트입니다
'''
plt.plot(x, y_pred, color='red', label='RANSAC')
'''
plt.plot() 함수는 선 그래프를 생성하는 함수입니다. 이 함수는 주어진 x와 y의 값에 따라 선을 그립니다. 일반적으로 plt.plot(x, y)의 형태로 사용되며, 여기서 x는 x축 값들을, y는 y축 값들을 나타냅니다
'''
plt.title("RANSAC MODEL")
plt.legend() #차트 상에 점이랑 선이 나올텐데 이게 어떤 선이고, 이게 어떠한 데이터 인지를 위에서 라벨링 한것을 차트 안에 결과 값을 나타내기 위해서 사용한것!
plt.show()