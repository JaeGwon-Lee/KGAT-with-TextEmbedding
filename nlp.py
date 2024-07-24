import argparse
import itertools
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util, models



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nlp',
                       default='keyword',
                       type=str,
                       help='Choose nlp method from {keyword, similarity, embedding}')
    parser.add_argument('--file_input',
                       default='melon_lyrics.csv',
                       type=str)
    return parser.parse_args()



# 가사 전처리
def preprocessing_lyrics(data) :
    
    lyrics = []
    for lyric_list in list(data['lyrics']) :

        if lyric_list == 'None' or lyric_list == 'Missing' or lyric_list == 'Error' :    # 가사가 없는 경우, 빈 리스트 출력
            lyrics.append('')
            continue
        lyric_list = eval(lyric_list)    # 리스트 활성화
        
        # 가사 데이터에서 노이즈 제거
        if lyric_list[0] == ' height:auto; 로 변경시, 확장됨 ' :
            lyric = ''
            for l in lyric_list[1:] :
                if lyric == '' :
                    l = l.strip().replace('\r', '').replace('\n', '').replace('\t', '').replace('`', "'")
                else :
                    l = l.strip().replace('`', "'")
                lyric = lyric + l
                if l != '' :
                    lyric = lyric + ' \n '
            lyric = lyric.rstrip(' \n ')    # 마지막 \n 삭제
        else :
            lyric = ''
            for l in lyric_list :
                l = l.strip().replace('`', "'")
                lyric = lyric + l
                if l != '' :
                    lyric = lyric + ' \n '
            lyric = lyric.rstrip(' \n ')
        
        lyrics.append(lyric)
        
    return lyrics



# 유사도 계산
def max_similarity(doc_embedding, candidate_embeddings, candidates, top_n):
    
    # 문서와 각 키워드들 간의 유사도
    distances = util.cos_sim(doc_embedding, candidate_embeddings)

    # 각 키워드들 간의 유사도
    distances_candidates = util.cos_sim(candidate_embeddings, candidate_embeddings)

    # 코사인 유사도에 기반하여 키워드들 중 상위 top_n개의 단어를 pick.
    words_idx = list(distances.argsort()[0][-top_n:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    return words_vals



# 키워드 추출
def get_keywords(lyrics) :
    
    model = SentenceTransformer('jhgan/ko-sroberta-multitask').to('cuda')
    okt = Okt()
    keywords = []

    for doc in tqdm(lyrics) :

        if len(doc) < 100 :    # 가사 길이가 너무 짧은 경우 제외
            doc = 'None'

        if doc == 'None' :    # 가사가 없는 경우, 빈 리스트 출력
            keywords.append([])
            continue

        # 한글은 명사만 출력
        tokenized_doc = okt.pos(doc, norm=True, stem=True)
        tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if (word[1] == 'Noun') or (word[1] == 'Alpha') or (word[0] == "'")])
        tokenized_nouns = tokenized_nouns.replace(" ' ", "'")

        # 영어는 불용어 제거
        try :
            count = CountVectorizer(stop_words='english').fit([tokenized_nouns])
            candidates = count.get_feature_names_out()
        except :
            keywords.append([])
            continue

        if len(candidates) < 5 :    # 추출된 단어가 5개 미만인 경우 제외
            keywords.append([])
            continue

        # 가사 전체와 단어 임베딩
        doc_embedding = model.encode([doc])
        candidate_embeddings = model.encode(candidates)

        # 유사도 계산
        keyword = max_similarity(doc_embedding, candidate_embeddings, candidates, top_n=5)
        keywords.append(keyword)

    return keywords



def get_high_similarity(lyrics, threshold=0.95):
    
    # 텍스트가 존재하는 데이터만 유사도 계산
    index_lyrictosong = dict()
    index_lyric = 0
    index_remove = []
    lyrics_selected = []

    for i, l in enumerate(lyrics):
        if l == '':
            index_remove.append(i)
            continue
        else :
            index_lyrictosong[index_lyric] = i
            lyrics_selected.append(l)
            index_lyric += 1
    
    model = SentenceTransformer('jhgan/ko-sroberta-multitask').to('cuda')
    paraphrases = util.paraphrase_mining(model, lyrics_selected, show_progress_bar=True, batch_size=128)
    
    # 유사도 threshold(default=0.95) 이상
    scores = []
    for score,_,_ in paraphrases:
        scores.append(score)
    n_para = np.where(np.array(scores)>=threshold)[0][-1]
    
    # 유사도 높은 텍스트 추출
    similarities = [[] for i in range(len(lyrics))]

    for score, i, j in paraphrases[:n_para+1]:
        i = index_lyrictosong[i]
        j = index_lyrictosong[j]
        similarities[i].append(j)
        similarities[j].append(i)

    for tmp_list in similarities:
        tmp_list.sort()
    
    return similarities



def get_embeddings(lyrics):
    
    # 텍스트가 존재하는 데이터만 임베딩 추출
    index_lyrictosong = dict()
    index_lyric = 0
    index_remove = []
    lyrics_selected = []

    for i, l in enumerate(lyrics):
        if l == '':
            index_remove.append(i)
            continue
        else :
            index_lyrictosong[index_lyric] = i
            lyrics_selected.append(l)
            index_lyric += 1
    
    model = SentenceTransformer('jhgan/ko-sroberta-multitask').to('cuda')
    
    # PCA를 통해 임베딩 차원을 768에서 256로 변경
    train_embeddings = model.encode(lyrics_selected, convert_to_numpy=True)
    new_dimension = 256
    pca = PCA(n_components=new_dimension)
    pca.fit(train_embeddings)
    pca_comp = np.asarray(pca.components_)
    dense = models.Dense(in_features=model.get_sentence_embedding_dimension(), out_features=new_dimension, bias=False, activation_function=nn.Identity())
    dense.linear.weight = nn.Parameter(torch.tensor(pca_comp))
    model.add_module('dense', dense)
    
    # 임베딩 추출
    embeddings_selected = model.encode(lyrics_selected, convert_to_numpy=True)

    embeddings = [np.zeros(256).tolist() for i in range(len(lyrics))]
    for i, embedding in enumerate(embeddings_selected):
        i = index_lyrictosong[i]
        embeddings[i] = embedding.tolist()
    
    return embeddings



def main() :
    
    args = parse_args()
    with open('datasets/melon/data/song_meta.json', encoding="utf-8") as f:
        songs = json.load(f)
    data = pd.read_csv(f'datasets/melon/data/{args.file_input}')
    lyrics = preprocessing_lyrics(data)
    
    if args.nlp == 'keyword':
        keywords = get_keywords(lyrics)
        keywords_list = sum(keywords, [])
        keywords_set = set(keywords_list)
        del_keywords = []
        for k in tqdm(keywords_set):
            if keywords_list.count(k) == 1:
                del_keywords.append(k)
        dict_ = [dict() for i in range(len(songs))]
        for song_num, keyword in zip(list(data['song_num']), keywords) :
            for k in keyword:
                if k in del_keywords:
                    keyword.remove(k)
            dict_[song_num]['keyword'] = keyword

    if args.nlp == 'similarity':
        similarities = get_high_similarity(lyrics)
        dict_ = [dict() for i in range(len(songs))]
        for song_num, similarity in zip(list(data['song_num']), similarities) :
            dict_[song_num]['similarity'] = similarity
    
    if args.nlp == 'embedding':
        embeddings = get_embeddings(lyrics)
        dict_ = [dict() for i in range(len(songs))]
        for song_num, embedding in zip(list(data['song_num']), embeddings) :
            dict_[song_num]['embedding'] = embedding

    with open(f'datasets/melon/data/{args.nlp}.json', 'w') as f:
        json.dump(dict_, f, ensure_ascii=False)



if __name__ == "__main__":
    main()