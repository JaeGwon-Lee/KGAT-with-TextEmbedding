import os
import sys
import random
from time import time
import json

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from model.KGAT import KGAT
from parser.parser_kgat import *
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_kgat import DataLoaderKGAT

from data_preprocessing.preprocessing_melon import preprocessing



'''
추천 성능 평가
'''
def evaluate(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]    # [[],[],[]] 형태 (내부 리스트 길이는 batch size)
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    cf_scores = []
    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}
    predict_dict = {}

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():
                batch_scores = model(batch_user_ids, item_ids, mode='predict')
                # (n_batch_users, n_items) 크기의 행렬 => batch_scores[user_idx][item_idx]로 해당 user, item에 대한 score 검색 가능

            batch_scores = batch_scores.cpu()
            batch_metrics, batch_predict = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks)
            predict_dict.update(batch_predict)
            cf_scores.append(batch_scores.numpy())
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)
    
    cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return cf_scores, metrics_dict, predict_dict



'''
Train
'''
def train(args):
    # seed 고정
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # logging (log_helper.py)
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # 데이터 생성
    if args.use_embedding == 1:
        entity_embed = preprocessing(n_user=args.n_user, nlp=args.nlp, use_embedding=args.use_embedding, data_size=args.data_size, logging=logging)    # entity BERT 임베딩 추가
        entity_embed = torch.tensor(entity_embed)
        embedding = None
    elif args.nlp == 'embedding':
        embedding = preprocessing(n_user=args.n_user, nlp=args.nlp, use_embedding=args.use_embedding, data_size=args.data_size, logging=logging)    # add BERT 임베딩
        embedding = torch.tensor(embedding)
        entity_embed = None
    else:
        preprocessing(n_user=args.n_user, nlp=args.nlp, use_embedding=args.use_embedding, data_size=args.data_size, logging=logging)
        entity_embed = None
        embedding = None
    
    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoader
    dataloader = DataLoaderKGAT(args, logging)

    # model & optimizer 구성
    if args.nlp == 'embedding':
        model = KGAT(args, dataloader.n_users, dataloader.n_entities, dataloader.n_relations, dataloader.A_in, entity_embed, embedding.to(device))
    else:
        model = KGAT(args, dataloader.n_users, dataloader.n_entities, dataloader.n_relations, dataloader.A_in, entity_embed, embedding)

    model.to(device)
    logging.info(model)

    cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # metrics 초기화
    best_epoch = -1
    best_recall = 0

    Ks = eval(args.Ks)    # metric@K에서의 K 리스트 (default=[20, 40, 60, 80, 100])
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in Ks}
    
    # 예측값 저장
    best_predict = {}

    # train model
    for epoch in range(1, args.n_epoch + 1):
        time0 = time()
        model.train()

        # train cf
        time1 = time()
        cf_total_loss = 0
        n_cf_batch = dataloader.n_cf_train // dataloader.cf_batch_size + 1

        for iter in range(1, n_cf_batch + 1):
            time2 = time()
            # 길이가 batch size로 동일한 user 텐서, pos item 텐서, neg item 텐서 샘플링
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = dataloader.generate_cf_batch(dataloader.train_user_dict, dataloader.cf_batch_size)
            cf_batch_user = cf_batch_user.to(device)
            cf_batch_pos_item = cf_batch_pos_item.to(device)
            cf_batch_neg_item = cf_batch_neg_item.to(device)

            cf_batch_loss = model(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, mode='train_cf')

            if np.isnan(cf_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (CF Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_cf_batch))
                sys.exit()

            cf_batch_loss.backward()
            cf_optimizer.step()
            cf_optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()

        #     if (iter % args.cf_print_every) == 0:
        #         logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), cf_total_loss / iter))
        # logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch))

        # train kg
        time3 = time()
        kg_total_loss = 0
        n_kg_batch = dataloader.n_kg_train // dataloader.kg_batch_size + 1

        for iter in range(1, n_kg_batch + 1):
            time4 = time()
            # 길이가 batch size로 동일한 head 텐서, relation 텐서, pos tail 텐서, neg tail 텐서 샘플링
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = dataloader.generate_kg_batch(dataloader.train_kg_dict, dataloader.kg_batch_size, dataloader.n_users_entities)
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            kg_batch_loss = model(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail, mode='train_kg')

            if np.isnan(kg_batch_loss.cpu().detach().numpy()):
                logging.info('ERROR (KG Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_kg_batch))
                sys.exit()

            kg_batch_loss.backward()
            kg_optimizer.step()
            kg_optimizer.zero_grad()
            kg_total_loss += kg_batch_loss.item()

            # if (iter % args.kg_print_every) == 0:
                # logging.info('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_kg_batch, time() - time4, kg_batch_loss.item(), kg_total_loss / iter))
        # logging.info('KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_kg_batch, time() - time3, kg_total_loss / n_kg_batch))

        # update attention
        time5 = time()
        h_list = dataloader.h_list.to(device)
        t_list = dataloader.t_list.to(device)
        r_list = dataloader.r_list.to(device)
        relations = list(dataloader.laplacian_dict.keys())
        model(h_list, t_list, r_list, relations, mode='update_att')
        # logging.info('Update Attention: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time5))

        logging.info('CF + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

        # evaluate cf
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time6 = time()
            _, metrics_dict, predict_dict = evaluate(model, dataloader, Ks, device)
            logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                epoch, time() - time6, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))

            epoch_list.append(epoch)
            for k in Ks:
                for m in ['precision', 'recall', 'ndcg']:
                    metrics_list[k][m].append(metrics_dict[k][m])
            best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], args.stopping_steps)

            # if should_stop:
            #     break

            if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch
                best_predict = predict_dict

    # save metrics
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx']
    for m in ['precision', 'recall', 'ndcg']:
        for k in Ks:
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append('{}@{}'.format(m, k))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + '/metrics.csv', index=False)

    # print best metrics
    best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
    logging.info('Best CF Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        int(best_metrics['epoch_idx']), best_metrics['precision@{}'.format(k_min)], best_metrics['precision@{}'.format(k_max)], best_metrics['recall@{}'.format(k_min)], best_metrics['recall@{}'.format(k_max)], best_metrics['ndcg@{}'.format(k_min)], best_metrics['ndcg@{}'.format(k_max)]))
    
    # 예측값 저장
    with open(args.save_dir + '/predict.json', 'w') as outfile:
        json.dump(best_predict, outfile)



if __name__ == '__main__':
    args = parse_kgat_args()
    train(args)
