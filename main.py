import argparse
import pickle
import time
from util import Data, split_validation
from model import *
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Grocery_and_Gourmet_Food', help='dataset name: 2019-Oct/Grocery_and_Gourmet_Food/Toys_and_Games')
parser.add_argument('--epoch', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--embSize', type=int, default=128, help='embedding size')
parser.add_argument('--num_heads', type=int, default=8, help='number of attention heads')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=float, default=2, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.2, help='price task maginitude')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')

opt = parser.parse_args()
print(opt)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
torch.cuda.set_device(1)

def main():
    # list[0]:session list[1]:label
    train_data = pickle.load(open('./data/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('./data/' + opt.dataset + '/test.txt', 'rb'))

    if opt.dataset == 'Grocery_and_Gourmet_Food':
        n_node = 6230
        n_price = 5
        n_category = 550
        n_brand = 1306
    elif opt.dataset == 'Toys_and_Games':
        n_node = 18979
        n_price = 5
        n_category = 430
        n_brand = 1429
    elif opt.dataset == '2019-Oct':
        n_node = 13026
        n_price = 10
        n_category = 226
        n_brand = 148
    else:
        print("unkonwn dataset")
    # data_formate: sessions, price_seq, matrix_session_item, matrix_session_price, matrix_pv, matrix_pb, matrix_pc, matrix_bv, matrix_bc, matrix_cv
    train_data = Data(train_data, shuffle=True, n_node=n_node, n_price=n_price, n_category=n_category, n_brand=n_brand)
    test_data = Data(test_data, shuffle=True, n_node=n_node, n_price=n_price, n_category=n_category, n_brand=n_brand)
    model = trans_to_cuda(
        DHCN(adjacency=train_data.adjacency, adjacency_pp=train_data.adjacency_pp, adjacency_cc=train_data.adjacency_cc,
             adjacency_bb=train_data.adjacency_bb, adjacency_vp=train_data.adjacency_vp,
             adjacency_vc=train_data.adjacency_vc, adjacency_vb=train_data.adjacency_vb,
             adjacency_pv=train_data.adjacency_pv, adjacency_pc=train_data.adjacency_pc,
             adjacency_pb=train_data.adjacency_pb, adjacency_cv=train_data.adjacency_cv,
             adjacency_cp=train_data.adjacency_cp, adjacency_cb=train_data.adjacency_cb,
             adjacency_bv=train_data.adjacency_bv, adjacency_bp=train_data.adjacency_bp,
             adjacency_bc=train_data.adjacency_bc, n_node=n_node, n_price=n_price, n_category=n_category,
             n_brand=n_brand, lr=opt.lr, layers=opt.layer, l2=opt.l2, beta=opt.beta, dataset=opt.dataset,
             num_heads=opt.num_heads, emb_size=opt.embSize, batch_size=opt.batchSize))

    top_K = [1, 5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['ndcg%d' % K] = np.mean(metrics['ndcg%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
            if best_results['metric%d' % K][2] < metrics['ndcg%d' % K]:
                best_results['metric%d' % K][2] = metrics['ndcg%d' % K]
                best_results['epoch%d' % K][2] = epoch
        print(metrics)
        # for K in top_K:
        #     print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tNDCG%d: %.4f\tEpoch: %d,  %d, %d' %
        #           (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],K, best_results['metric%d' % K][2],
        #            best_results['epoch%d' % K][0], best_results['epoch%d' % K][1], best_results['epoch%d' % K][2]))
        print('P@1\tP@5\tM@5\tN@5\tP@10\tM@10\tN@10\tP@20\tM@20\tN@20\t')
        print("%.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f\t %.2f" % (
            best_results['metric1'][0], best_results['metric5'][0], best_results['metric5'][1],
            best_results['metric5'][2], best_results['metric10'][0], best_results['metric10'][1],
            best_results['metric10'][2], best_results['metric20'][0], best_results['metric20'][1],
            best_results['metric20'][2]))
        print("%d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d" % (
            best_results['epoch1'][0], best_results['epoch5'][0], best_results['epoch5'][1],
            best_results['epoch5'][2], best_results['epoch10'][0], best_results['epoch10'][1],
            best_results['epoch10'][2], best_results['epoch20'][0], best_results['epoch20'][1],
            best_results['epoch20'][2]))

if __name__ == '__main__':
    main()
