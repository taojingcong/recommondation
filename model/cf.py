# coding: utf-8 -*-
import math
import pandas as pd


class UserCf:

    def __init__(self):
        self.file_path = 'data/test.csv'
        self._init_frame()
        self.action_interest = {'view':1,'cart':4,'remove_from_cart':-0.5,'purchase':8}

    def _init_frame(self):
        self.frame = pd.read_csv(self.file_path)

    @staticmethod
    def _cosine_sim(target_product, products):
        '''
        simple method for calculate cosine distance.
        e.g: x = [1 0 1 1 0], y = [0 1 1 0 1]
             cosine = (x1*y1+x2*y2+...) / [sqrt(x1^2+x2^2+...)+sqrt(y1^2+y2^2+...)]
             that means union_len(movies1, movies2) / sqrt(len(movies1)*len(movies2))
        '''
        #两个用户购买了多少相同的产品
        union_len = len(set(target_product['product_id']) & set(products['product_id']))
        #两个用户购买了多少相同品牌的产品
        target_brand_count = target_product.loc[:,'brand'].value_counts()
        len1 = float(len(target_product))
        len2 = float(len(products))
        part1 = {}
        part2 = {}

        for i,v in target_brand_count.items():
            part1[i] = float(v)/len1

        other_brand_count = products.loc[:,'brand'].value_counts()
        for i,v in other_brand_count.items():
            part2[i] = float(v)/len2

        for brand in part1:
            if brand in part2:
                union_len+=1.0/(abs(part1[brand]-part2[brand])+1)

        if union_len == 0: return 0.0
        product = len(target_product) * len(products)
        cosine = union_len / math.sqrt(product)
        return cosine

    def _get_top_n_users(self, target_user_id, top_n):
        '''
        calculate similarity between all users and return Top N similar users.
        '''
        #找到用户号为target_user_id所有记录
        #'user_id','category_id','product_id','event_type','brand'
        target_product = self.frame[self.frame['user_id'] == target_user_id][['user_id','category_id','product_id','event_type','brand']]
        #其他用户的Id
        other_users_id = [i for i in set(self.frame['user_id']) if i != target_user_id]
        #每个用户相关联的商品
        other_product = [self.frame[self.frame['user_id'] == i][['user_id','category_id','product_id','event_type','brand']] for i in other_users_id]

        sim_list = [self._cosine_sim(target_product, product) for product in other_product]
        sim_list = sorted(zip(other_users_id, sim_list), key=lambda x: x[1], reverse=True)
        return sim_list[:top_n]

    def _get_candidates_items(self, target_user_id):
        """
        Find all movies in source data and target_user did not meet before.
        """
        target_user_product = set(self.frame[self.frame['user_id'] == target_user_id]['product_id'])
        other_user_product = set(self.frame[self.frame['user_id'] != target_user_id]['product_id'])

        candidates_movies = list(target_user_product ^ other_user_product)
        return candidates_movies

    def _get_top_n_items(self, top_n_users, candidates_products, top_n):
        """
        calculate interest of candidates movies and return top n movies.
        e.g. interest = sum(sim * normalize_rating)
        """
        top_n_user_data = [self.frame[self.frame['user_id'] == k] for k, _ in top_n_users]
        interest_list = []
        #计算top_n_user对每个电影的评分
        for product_id in candidates_products:
            tmp = []
            #计算所有top_n_user对某个物品的兴趣
            brand = list(self.frame[self.frame['product_id'] == product_id]['brand'])[0]
            for user_data in top_n_user_data:
                #brand_count =
                if product_id in user_data['product_id'].values:
                    #tmp.append(user_data[user_data['MovieID'] == movie_id]['Rating'].values[0]/5)
                    user_product_interest = 0
                    action = user_data[user_data['product_id'] == product_id]
                    for index, row in action.iterrows():
                        user_product_interest+=self.action_interest[row['event_type']]
                    tmp.append(user_product_interest)
                else:
                    tmp.append(0)
            interest = sum([top_n_users[i][1] * tmp[i] for i in range(len(top_n_users))])
            interest_list.append((product_id, interest))
        interest_list = sorted(interest_list, key=lambda x: x[1], reverse=True)
        return interest_list[:top_n]

    def calculate(self, target_user_id=2, top_n=10):
        """
        user-cf for movies recommendation.
        """
        # most similar top n users
        top_n_users = self._get_top_n_users(target_user_id, top_n)
        # candidates movies for recommendation
        candidates_product = self._get_candidates_items(target_user_id)
        # most interest top n movies
        top_n_product = self._get_top_n_items(top_n_users, candidates_product, top_n=1)
        return top_n_product[0]
