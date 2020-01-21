## sklearn의 데이터세트

- 모형 실습을 위해서 예제 데이터 세트를 제공
- 3가지 분류
    - load  계열 : 이미 저장되어 있다 -> 용량이 작다
    - fetch 계열 : 인터넷에서 캐쉬 되어 다운로드 된다 -> 용량이 크다
    - make  계열 : 가상의 데이터 셋을 생성 -> 더미 데이터 생성

```python
import sklearn.datasets as mls
# 함수 목록
print(dir(mls))
```



### load() 계열

- 작은 데이터부터 큰 데이터까지 다양하다
- 작은 것들은 패키지에 이미 배포, 큰 것은 요청 시 다운로드해서 제공
- 종류
    - load_boston: 보스턴 집값 -> 회귀
    - load_breast_cancer: 유방암 -> 분류
    - load_diabetes: 당뇨병 -> 회귀
    - load_digits: 숫자 필기체(MNIST 계열) -> 분류
    - load_files
    - load_iris: 붓꽃 -> 분류
    - load_linnerud
    - load_sample_image
    - load_svmlight_file
    - load_svmlight_files
    - load_wine: 와인 -> 분류



### fetch() 계열

- 데이터가 커서 처음에는 설치가 안 됨
- 요청을 하면 다운 받아서 제공
- scikit_learn_data 밑으로 저장
- 종류
    - fetch_20newsgroups: 뉴스 그룹 텍스트 자료
    - fetch_20newsgroups_vectorized
    - fetch_california_housing
    - fetch_covtype: 토지 조사 자료 -> 회귀
    - fetch_kddcup99
    - fetch_lfw_pairs: 얼굴 이미지 자료
    - fetch_lfw_people: 얼굴 이미지 자료
    - fetch_mldata: ML 웹사이트 데이터
    - fetch_olivetti_faces: 얼굴 이미지 자료
    - fetch_openml
    - fetch_rcv1: 로이터 뉴스
    - fetch_species_distributions



### make 계열

- 모형 실험을 위해서 가상으로 만드는 데이터 세트
- 종류
    - make_regression : 회귀 분석용 데이터 세트
    - make_classification : 분류용 데이터 세트
    - make_blobs : 클러스터링용 가상 데이터 세트



```python
iris = mls.load_iris()
type(iris), dir(iris)
```

### sklearn.utils.Bunch

- sklearn에서 제공되는 데이터는 이형식을 대부분 따른다
- 변수
    - 'DESCR': 데이터에 대한 총괄적 설명
    - 'data': (필수) 독립 변수의 ndarray
    - 'target': (필수) 종속 변수의 ndarray
    - 'feature_names': (옵션) 독립 변수의 이름(컬럼)
    - 'target_names': (옵션) 종속 변수의 이름 리스트
    - 'filename' 