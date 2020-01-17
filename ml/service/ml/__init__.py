## __init__.py의 용도
# 1. python 3.3 이하 버전과 하위 호환을 위해서 사용
# 2. 패키지 자체를 지칭할 때 사용
from sklearn.externals import joblib
import json, re, urllib.request, sys, os

PI2=3.144
def a():
    print('구동')

if __name__!='__main__':
    # 0. 경로 -> 상수값 -> 환경변수 혹은 DB에서 획득
    MODEL_PATH = './ml/clf_model_202001161419.model'
    LABEL_PATH = './ml/clf_labels.label'

    # 1. 모델 로드(1회만)->요청이 많이지면 컨트롤이 가능한지 체크
    clf = joblib.load(MODEL_PATH)
    # 2. 레이블 로드
    with open(LABEL_PATH, 'r') as f:
        clf_label = json.load(f)

# 3. 예측 함수(input: 텍스트, output: 예측결과)
def detect_lang(text):
    # text -> 빈도계산 -> 알고리즘에 예측 요청(데이터 주입) -> 결과 리턴
    # A. 빈도계산
    text = text.lower()
    p = re.compile('[^a-z]')
    text = p.sub('', text)
    counts = [ 0 for n in range(26) ]
    limit_a = ord('a')
    for word in text:
        counts[ord(word)-limit_a] += 1
    total_count = sum(counts)
    freq = list(map(lambda x : x/total_count, counts))  # 정규화
    
    # B. 알고리즘에 예측 요청(데이터 주입)
    predict = clf.predict([freq])   # 입력 형태를 type 2차원 배열로 차원 맞춰줌
    na_code = predict[0]  # ex. 'en', 'fr' ...
    na_str = clf_label[na_code]

    # C. 결과 리턴
    return na_code, na_str

'''
POST방식
curl "https://openapi.naver.com/v1/papago/n2mt" \
-H "Content-Type: application/x-www-form-urlencoded; charset=UTF-8" \
-H "X-Naver-Client-Id: JUsT78p6Bg8nrZ3sHBxv" \
-H "X-Naver-Client-Secret: GX8hqnfdj2" \
-d "source=ko&target=en&text=만나서 반갑습니다." -v
'''

PAPAGO_URL = 'https://openapi.naver.com/v1/papago/n2mt'
CLIENT_ID = 'JUsT78p6Bg8nrZ3sHBxv'
SECRET_KEY = 'GX8hqnfdj2'

# 4. 번역 함수(현재: 파파고연동, 향후: RNN 구현)
def transfer_lang(text, na_input_code='en', na_output_code='ko'):
    print('파파고와 연동한 번역 처리 시작')

    # 한글의 URL 인코딩 처리
    encText = urllib.parse.quote(text) 
    data = f"source={na_input_code}&target={na_output_code}&text={encText}"
    request = urllib.request.Request(PAPAGO_URL)
    request.add_header("X-Naver-Client-Id",CLIENT_ID)
    request.add_header("X-Naver-Client-Secret",SECRET_KEY)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode == 200):
        return json.load(response)
    else:
        return {}

# 이 코드는 개발시 테스트했던 코드
# 의도(개발)시에만 작동해야 함
if __name__=='__main__':
    # print('테스트', PI2)
    test_str = "Le « nuage de mots-clés » est une sorte de condensé sémantique d'un document dans lequel les concepts clefs évoqués sont dotés d'une unité de taille (dans le sens du poids de la typographie utilisée) permettant de faire ressortir leur importance dans le site Web en cours ou dans les annuaires de sites utilisant ce même principe de fonctionnement. Il est possible de hiérarchiser ce système selon un ordre alphabétique de popularité ou encore de représentation dans le site en cours."
    transfer_lang(test_str)