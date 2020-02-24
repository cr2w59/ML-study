# 오차 함수

## 평균 제곱 계열

1. **Mean Squared Error: 평균 제곱 오차**
   - mean(square(실제값-예측값))
2. **Mean Absolute Error: 평균 절대 오차**
   - mean(abs(실제값-예측값))
3. **Mean Absolute Percentage Error: 평균 절대 백분율 오차**
   - mean(abs(실제값-예측값)/abs(실제값))
4. **Mean Squared Logarithmic Error: 평균 제곱 로그 오차**
   - mean(square((log(예측값)+1)-(log(실제값)+1)))



## 교차 엔트로피 계열

- ### Binary Crossentropy: 이항 교차 엔트로피

  - 두 개의 클래스 중에서 예측

  ------

  #### 피마 인디언의 당뇨병 예측

  1. *데이터 조사*

     - 샘플 수: 768
     - 속성: 8
       - 1: 과거 임신 횟수 / 2: 공복 혈당 농도 / 3: 확장기 혈압 / 4: 삼두근 피부 주름 두께
       - 5: 혈청 인슐린 / 6: 체질량 지수 / 7: 당뇨병 가족력 / 8: 나이
     - 클래스: 1
       - 1: 당뇨 / 0: 당뇨 아님

     ```python
     import pandas as pd
     
     df = pd.read_csv('./pima-indians-diabetes.csv', 
                      names=['pregnant', 'plasma', 'pressure', 'thickness',
                             'insulin', 'BMI', 'pedigree', 'age', 'class'])
     
     df.head(3)
     df.info()
     df.describe()
     ```

  2. *데이터 상관관계 파악*

     - Hitmap을 통해 class와 각 독립변수의 상관관계 파악

     ```python
     import matplotlib.pyplot as plt
     import seaborn as sns
     
     plt.figure(figsize=(12,12))
     sns.heatmap(df.corr(), linewidths=0.1, 
                 vmax=0.5, cmap=plt.cm.gist_heat, 
                 linecolor='white', annot=True)
     plt.show()
     ```

     ![11-3]()

     - class와 상관관계가 가장 큰 plasma만 따로 떼어 class와의 관계 재확인

     ```python
     grid = sns.FacetGrid(df, col='class')
     grid.map(plt.hist, 'plasma', bins=10)
     plt.show()
     ```

     ![11-4]()

  3. *데이터 가공*

     ```python
     X = df.iloc[:,0:8]
     Y = df.iloc[:,8]
     ```

  4. *모델 설계*

     - seed 고정
     - 순차적으로 model의 layer를 쌓음
       - model.add()로 새로운 층을 만들고 Dense()로 해당 층에 노드를 만듦

     ```python
     import numpy as np
     import tensorflow as tf
     from keras.models import Sequential
     from keras.layers import Dense
     
     SEED = 0
     np.random.seed(SEED)
     tf.set_random_seed(SEED)
     
     model = Sequential()
     # 8개의 입력값이 임의의 가중치를 가지고 12개의 노드로 전송돼 relu함수를 통과 
     model.add(Dense(12, input_dim=8, activation='relu'))	# 입력+은닉층
     # 이전 layer의 출력 값들을 입력 받아 8개의 노드로 전송돼 relu함수를 통과
     model.add(Dense(8, activation='relu'))	# 은닉층
     # 이전 layer의 출력 값들이 1개의 노드로 전송돼 sigmoid함수를 통과 후 최종 출력
     model.add(Dense(1, activation='sigmoid'))	# 출력층
     ```

     ![11-5]()

  5. *모델 컴파일*

     - 둘 중 하나를 결정하는 이항 분류 문제이므로 오차 함수로는 binary_crossentropy 사용
     - epochs: 전체 샘플의 총 반복 입력 횟수
     - batch_size: 한 번에 입력되는 입력 세트

     ```python
     model.compile(loss='binary_crossentropy', 
                   optimizer='adam', 
                   metrics=['accuracy'])
     model.fit(X, Y, epochs=200, batch_size=10)
     
     print('정확도: %.4f' % (model.evaluate(X, Y)[1]))
     ```

     ![10-2]()

     

- ### Categorical Crossentropy: 범주형 교차 엔트로피

  - 일반적 분류
  - 다중 분류

  ------

  #### Iris 품종 예측

  1. *데이터 조사*

     - 샘플 수: 150
     - 속성: 4
       - 1: 꽃받침 길이 / 2: 꽃받침 넓이 / 3: 꽃잎 길이 / 4: 꽃잎 넓이
     - 클래스: 1
       - Iris-setosa / Iris-versicolor/ Iris-virginica
       - 여러 개 중 하나를 고르는 **다중 분류**

     ```python
     import pandas as pd
     
     df = pd.read_csv('/content/iris.csv', 
                      names=['sepal_length', 'sepal_width', 
                             'petal_length', 'petal_width', 'species'])
     
     df.head(3)
     df.info()
     df.describe()
     ```

  2. *데이터 상관도 파악*

     - Pairplot을 통해 전체 데이터 한번에 파악

     ```python
     import matplotlib.pyplot as plt
     import seaborn as sns
     
     sns.pairplot(df, hue='species')
     plt.show()
     ```

     ![12-2]()

  3. *데이터 가공*

     - **원-핫 인코딩**: 여러 개의 범주형 값을 0과 1로만 이뤄진 형태로 바꿔 주는 기법

       - 문자열로 이루어져 있는 class(Y_obj)를 숫자 형태로 바꿔줘야 함

       - array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])인 Y가 array([1, 2, 3])으로 바뀜

         ```python
         X = df.iloc[:,0:4].astype(float)
         Y_obj = df.iloc[:,4]
         
         from sklearn.preprocessing import LabelEncoder
         
         e = LabelEncoder()
         e.fit(Y_obj)
         Y = e.transform(Y_obj)
         ```

       - 활성화 함수 적용을 위해 숫자 0과 1로 변형해야 함

       - array([1, 2, 3])인 Y가 다시 array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])로 바뀜

         ```python
         from keras.utils import np_utils
         
         Y_encoded = np_utils.to_categorical(Y)
         ```

  4. *모델 설계*

     - seed 고정
     - 순차적으로 model의 layer를 쌓음
       - 원-핫 레이블이 3개이므로 출력층의 노드는 3개
     - **softmax**함수를 통과하면 입력값들의 총합이 1이 됨
       - 큰 값이 두드러지게 되고 교차 엔트로피를 지나 1로 수렴

     ```python
     import numpy as np
     import tensorflow as tf
     from keras.models import Sequential
     from keras.layers import Dense
     
     SEED = 0
     np.random.seed(SEED)
     tf.set_random_seed(SEED)
     
     model = Sequential()
     # 4개의 입력값이 임의의 가중치를 가지고 16개의 노드로 전송돼 relu함수를 통과 
     model.add(Dense(16, input_dim=4, activation='relu'))	# 입력+은닉층
     # 이전 layer의 출력 값들이 3개의 노드로 전송돼 softmax함수를 통과 후 최종 출력
     model.add(Dense(3, activation='softmax'))	# 출력층
     ```

     ![12-3]()

  5. *모델 컴파일*

     - 다중 분류이므로 오차 함수로는 categorical_crossentropy 사용

     ```python
     model.compile(loss='categorical_crossentropy', 
                   optimizer='adam', 
                   metrics=['accuracy'])
     
     model.fit(X, Y_encoded, epochs=50, batch_size=1)
     
     print('정확도: %.4f' % (model.evaluate(X, Y_encoded)[1]))
     ```

