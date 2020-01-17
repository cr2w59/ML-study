import pymysql

def logBackup(na,tCode,oriTxt,tStc):
    connection = pymysql.connect(host='localhost', 
                            user='root', 
                            password='12341234', 
                            db='py_db',
                            charset='utf8mb4', 
                            cursorclass=pymysql.cursors.DictCursor)
    try:
        with connection.cursor() as cursor:
            sql = '''
            INSERT INTO `py_db`.`tbl_trans_lang_log` 
                (`oCode`, `tCode`, `oStr`, `tStc`) 
            VALUES
                (%s,%s,%s,%s);
            '''
            cursor.execute(sql,(na, tCode, oriTxt, tStc))
        connection.commit()
        print('로그 저장 완료')
    except Exception as e:
        print(e)
    finally:
        if connection:
            connection.close()