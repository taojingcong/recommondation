
predict_file = open('predict.csv','w',encoding='utf-8')
with open('predict1299 .csv','r',encoding='UTF-8') as f:
    for line in f.readlines():
        ID, Category, Pos_b, Pos_e, Privacy = line.split(',')
        predict_file.write(line)
        if int(ID)>=1299:
            break
with open('predict_1301_2600.csv','r',encoding='UTF-8') as f:
    for line in f.readlines():
        ID, Category, Pos_b, Pos_e, Privacy = line.split(',')
        if int(ID)>=1300 and int(ID)<1500:
            predict_file.write(line)
with open('bert_idcnn_crf-dymbert.csv','r',encoding='UTF-8') as f:
    for line in f.readlines():
        ID, Category, Pos_b, Pos_e, Privacy = line.split(',')
        if int(ID)>=1500:
            predict_file.write(line)
predict_file.close()