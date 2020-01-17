# 엔트리 포인트: 진입로, 시작점, 모든 경로법은 엔트리로부터 따짐
# 1. 모듈 가져오기
# flask 관련 모듈
from flask import Flask, render_template, request, jsonify, redirect
# 테스트 모듈
from ml.mod import *
from ml import PI2
# 언어 감지 및 번역 모듈
from ml import detect_lang, transfer_lang
# 로그 남기는 모듈
from db import logBackup

# 2. Flask 객체 생성
app = Flask( __name__ )
# 3. 라우팅
@app.route('/')
def home():
    text_str = '''
    Bong Joon-ho (Korean: 봉준호, Korean pronunciation: [poːŋ tɕuːnho → poːŋdʑunɦo]; born September 14, 1969) is a South Korean film director and screenwriter. He garnered international acclaim for his second feature film Memories of Murder (2003), before achieving commercial success with his subsequent films The Host (2006) and Snowpiercer (2013), both of which are among the highest-grossing films of all time in South Korea.
    '''
    print(detect_lang(text_str))
    return render_template('index.html')

# restful
@app.route('/bsgo', methods=['GET', 'POST'])
def bsgo():
    if request.method == 'GET':
        return render_template('bsgo.html')
    else:
        # 전달된 데이터 획득
        # print(request.form.get('o'))    # 만약 key 틀리면 None 반환
        # print(request.form['o'])    # 만약 key 틀리면 오류 발생
        oriTxt = request.form.get('o')  # 내용이 들어있고 100글자 이상
        
        # 언어감지
        na_code, na_str = detect_lang(oriTxt)

        # 결과 응답
        if na_code :
            res = { 'code':na_code, 'code_str':na_str }
        else :
            res = { 'code':'0', 'code_str':'언어 감지 실패' }
        return jsonify(res)

# 번역 처리
@app.route('/transfer', methods=['POST'])
def transfer():
    # 전달된 데이터 획득
    oriTxt = request.form.get('o')
    na = request.form.get('na')
    # 번역
    res = transfer_lang(oriTxt, na)
    # 로그 처리
    tCode = res['message']['result']['tarLangType']
    tStc = res['message']['result']['translatedText']
    logBackup(na,tCode,oriTxt,tStc)
    # 응답
    return jsonify(res)

# 4. 서버 가동
if __name__ == '__main__':
    app.run(debug=True)