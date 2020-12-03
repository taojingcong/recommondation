import pandas as pd


class Normal:
    def __init__(self):
        self.file_path = 'data/test.csv'
        self.action_interest = {'view':2,'cart':3,'remove_from_cart':-3.0,'purchase':2.5}
        self.frame = pd.read_csv(self.file_path)

    def predict(self,user_id):
        data = self.frame[self.frame['user_id'] == user_id]
        interest = {}
        has_buy = []
        for index, row in data.iterrows():
            if row['event_type']=='purchase':
                has_buy.append(row['product_id'])
        for index, row in data.iterrows():
            product_id = row['product_id']
            if product_id in has_buy:
                continue
            action = row['event_type']
            if product_id in interest:
                interest[product_id] += self.action_interest[action]
            else:
                interest[product_id] = self.action_interest[action]
        if len(interest) == 0:
            return has_buy[0]
        recommond = max(interest,key = interest.get)
        return recommond