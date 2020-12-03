from model.cf import UserCf
from model.normal import Normal
import sys
import pandas as pd
from tqdm import tqdm

if __name__=="__main__":
    arg = sys.argv[1]
    pd = pd.read_csv('data/test.csv')
    user_id = list(pd['user_id'])
    count = {}
    for i in range(len(pd)):
        if pd.iloc[i,2] not in count:
            count[pd.iloc[i,2]] = 1
        else:
            count[pd.iloc[i,2]] += 1
    max = 0
    max_id = 0
    for i in count:
        if count[i]>max:
            max_id = i
            max = count[i]
    print(max,max_id)


    prdict_csv = open('./data/predict.csv','w',encoding='UTF-8')
    prdict_csv.write("user_id,product_id\n")
    if arg == "cf":
        for id in tqdm(set(user_id)):
            product = UserCf().calculate(target_user_id=id)
            if product[1]==0:
                recommond = max_id
            else:
                recommond = product[0]
            prdict_csv.write(str(id)+','+str(recommond)+'\n')
    if arg == 'normal':
        for id in tqdm(set(user_id)):
            recommond = Normal().predict(user_id=id)
            print(recommond)
            prdict_csv.write(str(id)+','+str(recommond)+'\n')
    prdict_csv.close()

