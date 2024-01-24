#!/usr/bin/env python
# coding: utf-8

# In[2]:

import pandas as pd
import pyarrow
import json
from sklearn.metrics.pairwise import linear_kernel  # 두 벡터 간의 유사도 계산
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF를 기반 Vectorizer 객체 생성


mora = pd.read_csv('mora_3000.csv')  
mora['설명'].fillna('', inplace=True)


# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
tfidf_matrix = tfidf_vectorizer.fit_transform(
    mora['대분류'] + ' ' + mora['난이도'] + ' ' + mora['부위'] + ' ' + mora['도구'] + ' ' + mora['목적'] + ' ' + mora['설명'])

# 단어의 중요성 반영한 단어 목록 반환
terms = tfidf_vectorizer.get_feature_names_out()
df_tfidf = pd.DataFrame(data=tfidf_matrix.toarray(), columns=terms)

# 코사인 유사도(문서 유사도 산출 기법 중 하나) 계산
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# 운동 추천 함수
def get_recommend(data, exercise_movement, cosine_similarities):
    idx = data.index[data['운동 동작'] == exercise_movement].tolist()[0]  # 운동 동작 인덱스 찾기

    # 코사인 유사도 점수 정렬
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # 유사도 x[1] 기준
    sim_scores = sim_scores[1:11]  # 상위 10개 아이템 선택

    # 선택된 운동들의 인덱스 및 유사도 저장
    exercise_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]

    # 결과 저장 리스트 초기화
    result_list = []

    for exercise_index, similarity in zip(exercise_indices, similarity_scores):
        result = [
            data['운동 동작'].iloc[exercise_index],
            round(similarity, 4),
        ]

        json_result = json.dumps(result, ensure_ascii=False)
        print(json_result)

        result_list.append(result)

    return result_list

# 추천 받을 운동 동작 입력
exercise_movement = input("유사한 운동을 추천받을 동작을 입력하세요 ex)거꾸로 플랭크 : ")
recommendation_reason_list = get_recommend(mora, exercise_movement, cosine_similarities)
