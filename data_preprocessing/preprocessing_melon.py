import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util, models


def create_directory(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass



def preprocessing(n_user, nlp, use_embedding, data_size, logging):
    
    # 데이터 로드
    with open('datasets/melon/data/song_meta.json', encoding="utf-8") as f:
        songs = json.load(f)
    with open('datasets/melon/data/genre_gn_all.json', encoding="utf-8") as f:
        genres = json.load(f)
    with open('datasets/melon/data/train.json', encoding="utf-8") as f:
        train = json.load(f)
    if nlp == 'keyword' or nlp == 'similarity' or nlp=='embedding':
        with open(f'datasets/melon/data/{nlp}.json', encoding="utf-8") as f:
            texts = json.load(f)
    
    save_dir = f'datasets/melon/user{n_user}_{nlp}_{use_embedding}_{data_size}'
    
    ''' train.json '''
    
    # 설정한 user 수 만큼 train 딕셔너리{user: items} 생성 / user, item 리스트 생성
    user_set = set()
    item_set = set()
    train_dict = dict()
    
    if data_size == 'normal':
        u = 0
        while len(train_dict) < n_user:
            if len(train[u]['songs']) < 10:    # song이 10개 이상인 user만 사용
                u += 1
                continue
            user_set.add(train[u]['id'])
            item_set.update(train[u]['songs'])
            train_dict[train[u]['id']] = train[u]['songs']
            u += 1
    if data_size == 'small':
        u = 0
        while len(train_dict) < n_user:
            if (len(train[u]['songs']) < 10) or (len(train[u]['songs']) >= 30):    # song이 10개 이상, 30개 미만인 user만 사용
                u += 1
                continue
            user_set.add(train[u]['id'])
            item_set.update(train[u]['songs'])
            train_dict[train[u]['id']] = train[u]['songs']
            u += 1
    if data_size == 'big':
        u = 0
        while len(train_dict) < n_user:
            if len(train[u]['songs']) < 30:    # song이 30개 이상인 user만 사용
                u += 1
                continue
            user_set.add(train[u]['id'])
            item_set.update(train[u]['songs'])
            train_dict[train[u]['id']] = train[u]['songs']
            u += 1

    user_list = list(user_set)
    item_list = list(item_set)
    logging.info(f'num_user : {len(user_list)}')
    logging.info(f'num_item : {len(item_list)}')
    
    # 라벨링
    user_label = dict()
    label_user = dict()
    label = 0
    for i in user_list:
        user_label[int(i)] = label
        label_user[label] = int(i)
        label += 1

    item_label = dict()
    label_item = dict()
    label = 0
    for i in item_list:
        item_label[int(i)] = label
        label_item[label] = int(i)
        label += 1
    
    # 라벨링 저장
    create_directory(save_dir)
    create_directory(f'{save_dir}/label')
    with open(f'{save_dir}/label/user_label.json', 'w') as outfile:
        json.dump(user_label, outfile)
    with open(f'{save_dir}/label/label_user.json', 'w') as outfile:
        json.dump(label_user, outfile)
    with open(f'{save_dir}/label/item_label.json', 'w') as outfile:
        json.dump(item_label, outfile)
    with open(f'{save_dir}/label/label_item.json', 'w') as outfile:
        json.dump(label_item, outfile)
    
    # train-test 나누기
    np.random.seed(24)
    test_size = 0.2
    train_labeled = dict()
    test_labeled = dict()
    for key,value in train_dict.items():
        item_labeled = []
        for v in value:
            item_labeled.append(item_label[v])

        test_num = int(len(item_labeled)*test_size)    # test 개수
        np.random.shuffle(item_labeled)    # 랜덤화

        item_labeled_train = item_labeled[test_num:]
        item_labeled_test = item_labeled[:test_num]

        train_labeled[user_label[key]] = item_labeled_train
        test_labeled[user_label[key]] = item_labeled_test
    
    # train, test 저장
    with open(f'{save_dir}/train.json', 'w') as outfile:
        json.dump(train_labeled, outfile)
    with open(f'{save_dir}/test.json', 'w') as outfile:
        json.dump(test_labeled, outfile)
    
    ''' kg.txt '''
    
    # entity 종류별 저장
    song_list = []    # entity에 들어갈 item의 label은 item_label과 동일하게 설정
    artist_set = set()
    album_set = set()
    genre_set = set()
    
    for song in item_list:
        dict_tmp = songs[song]
        if dict_tmp['song_name'] == None:
            pass
        else:
            song_list.append(dict_tmp['song_name'])
        if dict_tmp['artist_name_basket'] == None:
            pass
        else:
            artist_set.update(dict_tmp['artist_name_basket'])
        if dict_tmp['album_name'] == None:
            pass
        else:
            album_set.add(dict_tmp['album_name'])
        if dict_tmp['song_gn_dtl_gnr_basket'] == None:
            pass
        else:
            for gnr in dict_tmp['song_gn_dtl_gnr_basket']:
                if gnr[-2:] == '01':    # "세부장르전체" -> 장르명
                    gnr = gnr[:-2]+'00'
                genre_set.add(genres[gnr])

    artist_list = list(artist_set)
    album_list = list(album_set)
    genre_list = list(genre_set)
    logging.info(f'num_artist : {len(artist_list)}')
    logging.info(f'num_album : {len(album_list)}')
    logging.info(f'num_genre : {len(genre_list)}')

    # 텍스트 부가 정보 추가 - keyword
    if nlp == 'keyword':
        text_set = set()
        for song in item_list:
            text_set.update(texts[song][nlp])
        text_list = list(text_set)
        logging.info(f'num_text : {len(text_list)}')
    
    # 텍스트 부가 정보 추가 - similarity
    if nlp == 'similarity':
        text_set = set()
        for song in item_list:
            text_list_tmp = texts[song][nlp]
            for text in text_list_tmp:
                if text not in item_set:    # 유사한 노래 리스트 중 item 리스트에 없는 노래 제거
                    continue
                text_set.add(text)
        text_list = list(text_set)
        logging.info(f'num_text : {len(text_list)}')
    
    # entity 라벨링
    entity_label = dict()
    label_entity = dict()
    entity_list = []
    
    label = 0
    for song in song_list:    # item도 entity 라벨링에 포함
        entity_label[song] = label
        label_entity[label] = song
        entity_list.append(song)
        label += 1
    for artist in artist_list:
        entity_label[artist] = label
        label_entity[label] = artist
        entity_list.append(artist)
        label += 1
    for album in album_list:
        entity_label[album] = label
        label_entity[label] = album
        entity_list.append(album)
        label += 1
    for genre in genre_list:
        entity_label[genre] = label
        label_entity[label] = genre
        entity_list.append(genre)
        label += 1
    if nlp == 'keyword' or nlp == 'similarity':
        for text in text_list:
            entity_label[text] = label
            label_entity[label] = text
            entity_list.append(text)
            label += 1
    
    # entity 라벨링 저장
    with open(f'{save_dir}/label/entity_label.json', 'w') as outfile:
        json.dump(entity_label, outfile)
    with open(f'{save_dir}/label/label_entity.json', 'w') as outfile:
        json.dump(label_entity, outfile)
    
    # knowledge graph 생성
    kg = []
    for i, song in enumerate(item_list):
        song_label = entity_label[entity_list[i]]
        
        dict_tmp = songs[song]
        if dict_tmp['artist_name_basket'] == None:
            pass
        else:
            # 아티스트
            for artist in dict_tmp['artist_name_basket']:
                kg.append({'h': song_label, 'r': 0, 't': entity_label[artist]})
        if dict_tmp['album_name'] == None:
            pass
        else:
            # 앨범
            kg.append({'h': song_label, 'r': 1, 't': entity_label[dict_tmp['album_name']]})
        if dict_tmp['song_gn_dtl_gnr_basket'] == None:
            pass
        else:
            # 장르
            for genre_number in dict_tmp['song_gn_dtl_gnr_basket']:
                if genre_number[-2:] == '01':    # "세부장르전체" -> 장르명
                    genre_number = genre_number[:-2]+'00'
                genre = genres[genre_number]
                kg.append({'h': song_label, 'r': 2, 't': entity_label[genre]})

        # 텍스트 - keyword
        if nlp == 'keyword':
            dict_tmp = texts[song]
            for text in dict_tmp[nlp]:
                kg.append({'h': song_label, 'r': 3, 't': entity_label[text]})
        # 텍스트 - similarity
        if nlp == 'similarity':
            dict_tmp = texts[song]
            for text in dict_tmp[nlp]:
                if text in text_set:
                    kg.append({'h': song_label, 'r': 3, 't': entity_label[text]})
    
    # kg 데이터프레임으로 저장
    kg_df = pd.DataFrame(kg)
    kg_df.to_csv(f'{save_dir}/kg.txt', sep=' ', index=False, header=False)
    
    # print('num_relation')
    for i in kg_df['r'].unique():
        logging.info(f'relation{i} : {len(kg_df[kg_df["r"]==i])}')
    
    # entity BERT 임베딩
    if use_embedding == 1:
        
        # 가사 임베딩 (entity_list에 가사를 넣는 것은 용량이 너무 큼 -> 가사 임베딩을 해놓은 파일에서 가져오기)
        if nlp == 'similarity':
            model = SentenceTransformer('jhgan/ko-sroberta-multitask')

            # PCA를 통해 임베딩 차원을 768에서 256로 변경
            train_embeddings = model.encode(entity_list[:-len(text_list)], convert_to_numpy=True)    # similarity entity는 제외하고 임베딩
            new_dimension = 256
            pca = PCA(n_components=new_dimension)
            pca.fit(train_embeddings)
            pca_comp = np.asarray(pca.components_)
            dense = models.Dense(in_features=model.get_sentence_embedding_dimension(), out_features=new_dimension, bias=False, activation_function=nn.Identity())
            dense.linear.weight = nn.Parameter(torch.tensor(pca_comp))
            model.add_module('dense', dense)

            # 임베딩 추출
            embeddings = model.encode(entity_list[:-len(text_list)], convert_to_numpy=True)
            
            # 가사 임베딩
            with open('datasets/melon/data/embedding.json', encoding="utf-8") as f:
                embedding_json = json.load(f)
            lyrics_embeddings = []
            for text in text_list:
                lyrics_embeddings.append(embedding_json[text]['embedding'])
            
            # 임베딩 합치기
            embeddings = np.array(embeddings.tolist() + lyrics_embeddings)

            return embeddings
        
        # textual entity 임베딩 (가사 임베딩을 제외한 일반적인 경우)
        else:
            model = SentenceTransformer('jhgan/ko-sroberta-multitask')

            # PCA를 통해 임베딩 차원을 768에서 256로 변경
            train_embeddings = model.encode(entity_list, convert_to_numpy=True)
            new_dimension = 256
            pca = PCA(n_components=new_dimension)
            pca.fit(train_embeddings)
            pca_comp = np.asarray(pca.components_)
            dense = models.Dense(in_features=model.get_sentence_embedding_dimension(), out_features=new_dimension, bias=False, activation_function=nn.Identity())
            dense.linear.weight = nn.Parameter(torch.tensor(pca_comp))
            model.add_module('dense', dense)

            # 임베딩 추출
            embeddings = model.encode(entity_list, convert_to_numpy=True)

            return embeddings
    
    # add BERT 임베딩
    if nlp == 'embedding':
        embeddings = []
        for song in item_list:
            embedding = texts[song]['embedding'] + list(np.zeros(64+32+16))
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        
        return embeddings
