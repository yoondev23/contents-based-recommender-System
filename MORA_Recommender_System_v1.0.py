#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.metrics.pairwise import linear_kernel   #선형 커널(두 벡터 사이의 점곱)을 사용하여 두 벡터 간의 유사도를 계산하는 함수 
from sklearn.feature_extraction.text import CountVectorizer # 단어의 등장 빈도를 기반으로하는 CountVectorizer 객체 생성
from sklearn.feature_extraction.text import TfidfVectorizer # TF-IDF를 기반으로 하는 TfidVectorizer 객체 생성


# In[3]:


# CSV 파일에서 데이터프레임 읽어오기
mora = pd.read_csv('mora_3000.csv')  

mora['설명'].fillna('', inplace=True)

mora.head(15)


# In[6]:


count_vectorizer = CountVectorizer(stop_words='english')
dtm_matrix = count_vectorizer.fit_transform(mora[['운동 동작', '대분류', '난이도', '부위', '도구', '목적', '설명']].apply(lambda x: ' '.join(x), axis=1))

terms = count_vectorizer.get_feature_names_out()
df_dtm = pd.DataFrame(data=dtm_matrix.toarray(), columns=terms)
print("DTM:")
print(df_dtm)


# In[4]:


# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(mora['대분류'] + ' ' + mora['부위'] + ' ' + mora['도구'] + ' ' + mora['목적']+ ' ' + mora['설명'])

# print(tfidf_matrix.toarray())

terms = tfidf_vectorizer.get_feature_names_out()
df_tfidf = pd.DataFrame(data=tfidf_matrix.toarray(), columns=terms)
print(df_tfidf)


# In[20]:


# 코사인 유사도 계산
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# 코사인 유사도를 데이터프레임으로 변환하여 출력
doc_names = mora.index
cosine_sim_df = pd.DataFrame(cosine_similarities, index=doc_names, columns=doc_names)
print("코사인 유사도 행렬:")
print(cosine_sim_df)


# In[14]:


# 운동 추천 함수
def 추천받기(운동_동작):
    idx = mora.index[mora['운동 동작'] == 운동_동작].tolist()[0]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:10]  # 상위 7개 아이템 선택 
    exercise_indices = [i[0] for i in sim_scores]
    similar_exercises = mora['운동 동작'].iloc[exercise_indices]
    similarity_scores = [i[1] for i in sim_scores]
    reasons = []

    print(f"\n\"{운동_동작}\"에 대한 정보:")
    print(f"  - 대분류: {mora['대분류'].iloc[idx]}")
#     print(f"  - 난이도: {mora['난이도'].iloc[idx]}")
    print(f"  - 부위: {mora['부위'].iloc[idx]}")
    print(f"  - 도구: {mora['도구'].iloc[idx]}")
    print(f"  - 목적: {mora['목적'].iloc[idx]}")
    print(f"  - 설명: {mora['설명'].iloc[idx]}")

    print(f"\n\"추천하는 운동 {운동_동작}\"와 유사한 운동:")
    for exercise_index, similarity in zip(exercise_indices, similarity_scores):
        reasons.append({
            '운동 동작': mora['운동 동작'].iloc[exercise_index],
            '유사도': similarity,
            '이유': {
                '대분류': mora['대분류'].iloc[exercise_index],
                '난이도': mora['난이도'].iloc[exercise_index],
                '부위': mora['부위'].iloc[exercise_index],
                '도구': mora['도구'].iloc[exercise_index],
                '목적': mora['목적'].iloc[exercise_index],
                '설명': mora['설명'].iloc[exercise_index],
            }
        })
        print(f"  - {mora['운동 동작'].iloc[exercise_index]} (유사도: {similarity:.2f})")

    return reasons


# In[15]:


# 추천 받을 운동 동작 선택
운동_동작 = '거꾸로 플랭크'
추천이유목록 = 추천받기(운동_동작)

# 결과 출력
for 추천이유 in 추천이유목록:
    print(f"\n\"{추천이유['운동 동작']}\"에 대한 정보:")
    print(f"  - 대분류: {추천이유['이유']['대분류']}")
#     print(f"  - 난이도: {추천이유['이유']['난이도']}")
    print(f"  - 부위: {추천이유['이유']['부위']}")
    print(f"  - 도구: {추천이유['이유']['도구']}")
    print(f"  - 목적: {추천이유['이유']['목적']}")
    print(f"  - 설명: {추천이유['이유']['설명']}")
    print(f"  - 유사도: {추천이유['유사도']:.2f}")


# In[ ]:





# In[ ]:




