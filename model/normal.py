import pandas as pd


class Normal:
    def __init__(self):
        self.file_path = 'data/test.csv'
        self.action_interest = {'view':2,'cart':3,'remove_from_cart':-1.0,'purchase':2.5}
        self.frame = pd.read_csv(self.file_path)

    def predict(self,user_id):
        data = self.frame[self.frame['user_id'] == user_id]
        interest = {}
        for index, row in data.iterrows():
            product_id = row['product_id']
            action = row['event_type']
            if product_id in interest:
                interest[product_id] += self.action_interest[action]
            else:
                interest[product_id] = self.action_interest[action]
        recommond = max(interest,key = interest.get)
        return recommond