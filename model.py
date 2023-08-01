import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class HyperConv(Module):
    def __init__(self, layers, dataset, emb_size, n_node, n_price, n_category, n_brand):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset
        self.n_node = n_node
        self.n_price = n_price
        self.n_category = n_category
        self.n_brand = n_brand

        self.w_pv = nn.Linear(self.emb_size, self.emb_size)
        self.w_bv = nn.Linear(self.emb_size, self.emb_size)
        self.w_cv = nn.Linear(self.emb_size, self.emb_size)

        self.w_bp = nn.Linear(self.emb_size, self.emb_size)
        self.w_cp = nn.Linear(self.emb_size, self.emb_size)
        self.w_vp = nn.Linear(self.emb_size, self.emb_size)

        self.w_pb = nn.Linear(self.emb_size, self.emb_size)
        self.w_cb = nn.Linear(self.emb_size, self.emb_size)
        self.w_vb = nn.Linear(self.emb_size, self.emb_size)

        self.w_pc = nn.Linear(self.emb_size, self.emb_size)
        self.w_bc = nn.Linear(self.emb_size, self.emb_size)
        self.w_vc = nn.Linear(self.emb_size, self.emb_size)


        self.tran_pv = nn.Linear(self.emb_size, self.emb_size)
        self.tran_pc = nn.Linear(self.emb_size, self.emb_size)
        self.tran_pb = nn.Linear(self.emb_size, self.emb_size)

        self.tran_cv = nn.Linear(self.emb_size, self.emb_size)
        self.tran_cp = nn.Linear(self.emb_size, self.emb_size)
        self.tran_cb = nn.Linear(self.emb_size, self.emb_size)

        self.tran_bv = nn.Linear(self.emb_size, self.emb_size)
        self.tran_bp = nn.Linear(self.emb_size, self.emb_size)
        self.tran_bc = nn.Linear(self.emb_size, self.emb_size)

        self.tran_pv2 = nn.Linear(self.emb_size, self.emb_size)
        self.tran_pc2 = nn.Linear(self.emb_size, self.emb_size)
        self.tran_pb2 = nn.Linear(self.emb_size, self.emb_size)

        self.tran_cv2 = nn.Linear(self.emb_size, self.emb_size)
        self.tran_cp2 = nn.Linear(self.emb_size, self.emb_size)
        self.tran_cb2 = nn.Linear(self.emb_size, self.emb_size)

        self.tran_bv2 = nn.Linear(self.emb_size, self.emb_size)
        self.tran_bp2 = nn.Linear(self.emb_size, self.emb_size)
        self.tran_bc2 = nn.Linear(self.emb_size, self.emb_size)

        self.a_pb = nn.Linear(self.emb_size, self.emb_size)
        self.a_pc = nn.Linear(self.emb_size, self.emb_size)
        self.a_bc = nn.Linear(self.emb_size, self.emb_size)
        self.a_bp = nn.Linear(self.emb_size, self.emb_size)
        self.a_cb = nn.Linear(self.emb_size, self.emb_size)
        self.a_cp = nn.Linear(self.emb_size, self.emb_size)

        self.b_pb = nn.Linear(self.emb_size, self.emb_size)
        self.b_pc = nn.Linear(self.emb_size, self.emb_size)
        self.b_bc = nn.Linear(self.emb_size, self.emb_size)
        self.b_bp = nn.Linear(self.emb_size, self.emb_size)
        self.b_cb = nn.Linear(self.emb_size, self.emb_size)
        self.b_cp = nn.Linear(self.emb_size, self.emb_size)

        # self.mat_v = nn.Parameter(torch.Tensor(self.n_node, self.emb_size))
        self.mat_pc = nn.Parameter(torch.Tensor(self.n_price, 1))
        self.mat_pb = nn.Parameter(torch.Tensor(self.n_price, 1))
        self.mat_pv = nn.Parameter(torch.Tensor(self.n_price, 1))

        self.mat_cp = nn.Parameter(torch.Tensor(self.n_category, 1))
        self.mat_cb = nn.Parameter(torch.Tensor(self.n_category, 1))
        self.mat_cv = nn.Parameter(torch.Tensor(self.n_category, 1))

        self.mat_bp = nn.Parameter(torch.Tensor(self.n_brand, 1))
        self.mat_bc = nn.Parameter(torch.Tensor(self.n_brand, 1))
        self.mat_bv = nn.Parameter(torch.Tensor(self.n_brand, 1))

        # self.mat_bc = nn.Parameter(torch.Tensor(self.n_brand, self.emb_size))

        self.a_i_g = nn.Linear(self.emb_size, self.emb_size)
        self.b_i_g = nn.Linear(self.emb_size, self.emb_size)

        self.w_v_1 = nn.Linear(self.emb_size * 4, self.emb_size)
        self.w_v_11 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.w_v_2 = nn.Linear(self.emb_size * 1, self.emb_size)
        self.w_v_3 = nn.Linear(self.emb_size * 1, self.emb_size)
        self.w_v_4 = nn.Linear(self.emb_size * 1, self.emb_size)

        self.w_p_1 = nn.Linear(self.emb_size * 4, self.emb_size)
        self.w_p_11 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.w_p_2 = nn.Linear(self.emb_size * 1, self.emb_size)
        self.w_p_3 = nn.Linear(self.emb_size * 1, self.emb_size)
        self.w_p_4 = nn.Linear(self.emb_size * 1, self.emb_size)

        self.w_c_1 = nn.Linear(self.emb_size * 4, self.emb_size)
        self.w_c_11 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.w_c_2 = nn.Linear(self.emb_size * 1, self.emb_size)
        self.w_c_3 = nn.Linear(self.emb_size * 1, self.emb_size)
        self.w_c_4 = nn.Linear(self.emb_size * 1, self.emb_size)

        self.w_b_1 = nn.Linear(self.emb_size * 4, self.emb_size)
        self.w_b_11 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.w_b_2 = nn.Linear(self.emb_size * 1, self.emb_size)
        self.w_b_3 = nn.Linear(self.emb_size * 1, self.emb_size)
        self.w_b_4 = nn.Linear(self.emb_size * 1, self.emb_size)

        self.dropout10 = nn.Dropout(0.1)
        self.dropout20 = nn.Dropout(0.2)
        self.dropout30 = nn.Dropout(0.3)
        self.dropout40 = nn.Dropout(0.4)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout60 = nn.Dropout(0.6)
        self.dropout70 = nn.Dropout(0.7)
        self.dropout90 = nn.Dropout(0.9)

    def forward(self, adjacency, adjacency_pp, adjacency_cc, adjacency_bb, adjacency_vp, adjacency_vc, adjacency_vb,
                adjacency_pv, adjacency_pc, adjacency_pb, adjacency_cv, adjacency_cp, adjacency_cb, adjacency_bv,
                adjacency_bp, adjacency_bc, embedding, pri_emb, cate_emb, bra_emb):
        for i in range(self.layers):
            item_embeddings = self.inter_gate3(
                self.w_v_1, self.w_v_11, self.w_v_2, self.w_v_3, self.w_v_4, embedding,
                self.get_embedding(adjacency_vp, pri_emb),
                self.get_embedding(adjacency_vc, cate_emb),
                self.get_embedding(adjacency_vb, bra_emb)) + self.get_embedding(adjacency, embedding)

            price_embeddings = self.inter_gate3(
                self.w_p_1, self.w_p_11, self.w_p_2, self.w_p_3, self.w_p_4, pri_emb,
                self.intra_gate2(adjacency_pv, self.mat_pv, self.tran_pv, self.tran_pv2, pri_emb, embedding),
                self.intra_gate2(adjacency_pc, self.mat_pc, self.tran_pc, self.tran_pc2, pri_emb, cate_emb),
                self.intra_gate2(adjacency_pb, self.mat_pb, self.tran_pb, self.tran_pb2, pri_emb,
                                 bra_emb)) + self.get_embedding(adjacency_pp, pri_emb)

            category_embeddings = self.inter_gate3(
                self.w_c_1, self.w_c_11, self.w_c_2, self.w_c_3, self.w_c_4, cate_emb,
                self.intra_gate2(adjacency_cp, self.mat_cp, self.tran_cp, self.tran_cp2, cate_emb, pri_emb),
                self.intra_gate2(adjacency_cv, self.mat_cv, self.tran_cv, self.tran_cv2, cate_emb, embedding),
                self.intra_gate2(adjacency_cb, self.mat_cb, self.tran_cb, self.tran_cb2, cate_emb,
                                 bra_emb)) + self.get_embedding(adjacency_cc, cate_emb)
            brand_embeddings = self.inter_gate3(
                self.w_b_1, self.w_b_11, self.w_b_2, self.w_b_3, self.w_b_4, bra_emb,
                self.intra_gate2(adjacency_bp, self.mat_bp, self.tran_bp, self.tran_bp2, bra_emb, pri_emb),
                self.intra_gate2(adjacency_bc, self.mat_bc, self.tran_bc, self.tran_bc2, bra_emb, cate_emb),
                self.intra_gate2(adjacency_bv, self.mat_bv, self.tran_bv, self.tran_bv2, bra_emb, embedding)
            ) + self.get_embedding(adjacency_bb, bra_emb)
            embedding = item_embeddings
            pri_emb = price_embeddings
            cate_emb = category_embeddings
            bra_emb = brand_embeddings

        return item_embeddings, price_embeddings

    def get_embedding(self, adjacency, embedding):
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        embs = embedding
        item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), embs)
        return item_embeddings
    def intra_gate(self, adjacency, mat_v, trans1, trans2, embedding1, embedding2):
        # v_attention to get embedding of type, and then gate to get final type embedding
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        matrix = adjacency.to_dense().cuda()
        # tran_emb2 = trans1(embedding2)
        mat_v = mat_v.expand(mat_v.shape[0], self.emb_size)
        alpha = torch.mm(mat_v, torch.transpose(embedding2, 1, 0))
        alpha = torch.nn.Softmax(dim=1)(alpha)
        alpha = alpha * matrix
        sum_alpha_row = torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + 1e-8
        alpha = alpha / sum_alpha_row

        type_embs = torch.mm(alpha, embedding2)

        item_embeddings = type_embs
        return self.dropout70(item_embeddings)
    def intra_gate2(self, adjacency, mat_v, trans1, trans2, embedding1, embedding2):
        # v_attention to get embedding of type, and then gate to get final type embedding
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        matrix = adjacency.to_dense().cuda()

        tran_emb2 = trans1(embedding2)

        alpha = torch.mm(embedding1, torch.transpose(tran_emb2, 1, 0))
        alpha = torch.nn.Softmax(dim=1)(alpha)
        alpha = alpha * matrix
        sum_alpha_row = torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + 1e-8
        alpha = alpha / sum_alpha_row

        type_embs = torch.mm(alpha, embedding2)
        item_embeddings = type_embs
        return item_embeddings
    def inter_gate(self, a_o_g, b_o_g1, b_o_g2, emb_mat1, emb_mat2, emb_mat3):
        all_emb1 = self.dropout70(torch.cat([emb_mat1, emb_mat2, emb_mat3], 1))
        # all_emb2 = torch.cat([emb_mat1, emb_mat3], 1)
        gate1 = torch.sigmoid(a_o_g(all_emb1) + b_o_g1(emb_mat2) + b_o_g2(emb_mat3))
        # gate2 = torch.sigmoid(a_o_g(all_emb2) + b_o_g2(emb_mat3))
        h_embedings = emb_mat1 + gate1 * emb_mat2 + (1 - gate1) * emb_mat3


        return h_embedings

    def inter_gate3(self, w1, w11, w2, w3, w4, emb1, emb2, emb3, emb4):
        # 4 to 1
        all_emb = torch.cat([emb1, emb2, emb3, emb4], 1)

        gate1 = torch.tanh(w1(all_emb) + w2(emb2))
        gate2 = torch.tanh(w1(all_emb) + w3(emb3))
        gate3 = torch.tanh(w1(all_emb) + w4(emb4))
        # gate2 = torch.sigmoid(a_o_g(all_emb2) + b_o_g2(emb_mat3))
        h_embedings = emb1 + gate1 * emb2 + gate2 * emb3 + gate3 * emb4
        return h_embedings


    def intra_att(self, adjacency, trans1, trans2, embedding1, embedding2):
        # mlp 映射到相同空间，然后计算cosine相似度，确定attention值
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        tran_m1 = trans1(embedding1)
        tran_m2 = trans2(embedding2)
        tran_m2 = torch.transpose(tran_m2, 1, 0)
        atten = torch.matmul(tran_m1, tran_m2)
        sum2_m1 = torch.sum(torch.mul(tran_m1, tran_m1),1)
        sum2_m2 = torch.sum(torch.mul(torch.transpose(tran_m2, 1, 0), torch.transpose(tran_m2, 1, 0)),1)
        fenmu_m1 = torch.sqrt(sum2_m1).unsqueeze(1)
        fenmu_m2 = torch.sqrt(sum2_m2).unsqueeze(1)
        fenmu = torch.matmul(fenmu_m1, torch.transpose(fenmu_m2, 1, 0)) + 1e-8
        atten = torch.div(atten, fenmu)
        atten = torch.nn.Softmax(dim=1)(atten)
        atten = torch.mul(adjacency.to_dense().cuda(),atten) + 1e-8
        atten = torch.div(atten, torch.sum(atten, 1).unsqueeze(1).expand_as(atten))
        embs = embedding2
        item_embeddings = torch.mm(atten, embs)
        return item_embeddings

    def inter_att(self, v_mat, emb_mat1, emb_mat2, emb_mat3=None):
        if emb_mat3 is not None:
            alpha1 = torch.mul(v_mat, emb_mat1)
            alpha1 = torch.sum(alpha1, 1).unsqueeze(1)
            alpha2 = torch.mul(v_mat, emb_mat2)
            alpha2 = torch.sum(alpha2, 1).unsqueeze(1)
            alpha3 = torch.mul(v_mat, emb_mat3)
            alpha3 = torch.sum(alpha3, 1).unsqueeze(1)
            alpha = torch.cat([alpha1, alpha2, alpha3], 1)
            alpha = torch.nn.Softmax(dim=1)(alpha).permute(1, 0)
            alpha1 = alpha[0]
            alpha2 = alpha[1]
            alpha3 = alpha[2]
            weight_embs = alpha1.unsqueeze(1).expand_as(emb_mat1) * emb_mat1 + alpha2.unsqueeze(1).expand_as(
                emb_mat2) * emb_mat2 + alpha3.unsqueeze(1).expand_as(emb_mat3) * emb_mat3
        else:
            alpha1 = torch.mul(v_mat, emb_mat1)
            alpha1 = torch.sum(alpha1, 1).unsqueeze(1)
            alpha2 = torch.mul(v_mat, emb_mat2)
            alpha2 = torch.sum(alpha2, 1).unsqueeze(1)
            # alpha3 = torch.mul(v_mat, emb_mat3)
            # alpha3 = torch.sum(alpha3, 1).unsqueeze(1)
            alpha = torch.cat([alpha1, alpha2], 1)
            alpha = torch.nn.Softmax(dim=1)(alpha).permute(1, 0)
            alpha1 = alpha[0]
            alpha2 = alpha[1]
            # alpha3 = alpha[2]
            weight_embs = alpha1.unsqueeze(1).expand_as(emb_mat1) * emb_mat1 + alpha2.unsqueeze(1).expand_as(
                emb_mat2) * emb_mat2

        return weight_embs

class DHCN(Module):
    def __init__(self, adjacency, adjacency_pp, adjacency_cc, adjacency_bb, adjacency_vp, adjacency_vc, adjacency_vb, adjacency_pv, adjacency_pc, adjacency_pb, adjacency_cv, adjacency_cp, adjacency_cb, adjacency_bv, adjacency_bp, adjacency_bc, n_node, n_price, n_category, n_brand, lr, layers, l2, beta, dataset, num_heads=4, emb_size=100, batch_size=100):
        super(DHCN, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.n_price = n_price
        self.n_category = n_category
        self.n_brand = n_brand
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta

        self.adjacency = adjacency
        self.adjacency_pp = adjacency_pp
        self.adjacency_cc = adjacency_cc
        self.adjacency_bb = adjacency_bb

        self.adjacency_vp = adjacency_vp
        self.adjacency_vc = adjacency_vc
        self.adjacency_vb = adjacency_vb

        self.adjacency_pv = adjacency_pv
        self.adjacency_pc = adjacency_pc
        self.adjacency_pb = adjacency_pb

        self.adjacency_cv = adjacency_cv
        self.adjacency_cp = adjacency_cp
        self.adjacency_cb = adjacency_cb

        self.adjacency_bv = adjacency_bv
        self.adjacency_bp = adjacency_bp
        self.adjacency_bc = adjacency_bc

        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.price_embedding = nn.Embedding(self.n_price, self.emb_size)
        self.category_embedding = nn.Embedding(self.n_category, self.emb_size)
        self.brand_embedding = nn.Embedding(self.n_brand, self.emb_size)


        self.pos_embedding = nn.Embedding(2000, self.emb_size)
        self.HyperGraph = HyperConv(self.layers, dataset, self.emb_size, self.n_node, self.n_price, self.n_category, self.n_brand)
        self.w_1 = nn.Linear(self.emb_size*2, self.emb_size)
        self.w_price_1 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.w_2 = nn.Linear(self.emb_size, 1)
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.glu3 = nn.Linear(self.emb_size, self.emb_size, bias=False)

        # self_attention
        if emb_size % num_heads != 0:  # 整除
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (emb_size, num_heads))
            # 参数定义
        self.num_heads = num_heads  # 4
        self.attention_head_size = int(emb_size / num_heads)  # 16
        self.all_head_size = int(self.num_heads * self.attention_head_size)
        # query, key, value
        self.query = nn.Linear(self.emb_size , self.emb_size )  # 128, 128
        self.key = nn.Linear(self.emb_size , self.emb_size )
        self.value = nn.Linear(self.emb_size , self.emb_size )

        self.w_p_z = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_p_r = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_p = nn.Linear(self.emb_size, self.emb_size, bias=True)

        self.u_i_z = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_i_r = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_i = nn.Linear(self.emb_size, self.emb_size, bias=True)

        # gate
        self.w_pi_1 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_pi_2 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_c_z = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_j_z = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_c_r = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_j_r = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_p = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_p = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.w_i = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.u_i = nn.Linear(self.emb_size, self.emb_size, bias=True)

        # multi-task merge
        self.merge_w = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w1 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w2 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w3 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w4 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w5 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w6 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w7 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w8 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w9 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w10 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w11 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.merge_w12 = nn.Linear(self.emb_size, self.emb_size, bias=True)

        self.mlp_m_p_1 =  nn.Linear(self.emb_size*2, self.emb_size, bias=True)
        self.mlp_m_i_1 = nn.Linear(self.emb_size * 2, self.emb_size, bias=True)

        self.mlp_m_p_2 = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.mlp_m_i_2 = nn.Linear(self.emb_size, self.emb_size, bias=True)

        self.dropout = nn.Dropout(0.2)
        self.emb_dropout = nn.Dropout(0.25)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout7 = nn.Dropout(0.7)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_sess_emb(self, item_embedding, price_embedding, session_item, price_seqs, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        mask = mask.float().unsqueeze(-1)

        price_embedding = torch.cat([zeros, price_embedding], 0)
        get_pri = lambda i: price_embedding[price_seqs[i]]
        seq_pri = torch.cuda.FloatTensor(self.batch_size, list(price_seqs.shape)[1], self.emb_size).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        for i in torch.arange(price_seqs.shape[0]):
            seq_pri[i] = get_pri(i)

        # add position to price emebedding
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        len = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)
        seq_pri = self.w_price_1(torch.cat([pos_emb, seq_pri], -1))


        # self-attention to get price preference
        attention_mask = mask.permute(0,2,1).unsqueeze(1)  # [bs, 1, 1, seqlen]
        attention_mask = (1.0 - attention_mask) * -10000.0

        mixed_query_layer = self.query(seq_pri)  # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(seq_pri)  # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(seq_pri)  # [bs, seqlen, hid_size]

        attention_head_size = int(self.emb_size / self.num_heads)
        query_layer = self.transpose_for_scores(mixed_query_layer, attention_head_size)  # [bs, 8, seqlen, 16]
        key_layer = self.transpose_for_scores(mixed_key_layer, attention_head_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, attention_head_size)  # [bs, 8, seqlen, 16]
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(attention_head_size)  # [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, 8, seqlen, seqlen]
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # 矩阵相乘，[bs, 8, seqlen, seqlen]*[bs, 8, seqlen, 16] = [bs, 8, seqlen, 16]
        context_layer = torch.matmul(attention_probs, value_layer)  # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, seqlen, 8, 16]
        new_context_layer_shape = context_layer.size()[:-2] + (self.emb_size,)  # [bs, seqlen, 128]
        sa_result = context_layer.view(*new_context_layer_shape)
        # last hidden state as price preferences
        item_pos = torch.tensor(range(1, seq_pri.size()[1] + 1), device='cuda')
        item_pos = item_pos.unsqueeze(0).expand_as(price_seqs)

        item_pos = item_pos * mask.squeeze(2)
        item_last_num = torch.max(item_pos, 1)[0].unsqueeze(1).expand_as(item_pos)
        last_pos_t = torch.where(item_pos - item_last_num >= 0, torch.tensor([1.0], device='cuda'),
                                 torch.tensor([0.0], device='cuda'))
        last_interest = last_pos_t.unsqueeze(2).expand_as(sa_result) * sa_result
        price_pre = torch.sum(last_interest, 1)
        # average as price preferences

        # price_pre = torch.div(torch.sum(sa_result, 1), session_len)

        item_embedding = torch.cat([zeros, item_embedding], 0)
        # get = lambda i: item_embedding[session_item[i]]
        # seq_h = torch.cuda.FloatTensor(self.batch_size, list(session_item.shape)[1], self.emb_size).fill_(0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)

        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        # last
        last_price = last_pos_t.unsqueeze(2).expand_as(seq_h) * seq_h
        hl = torch.sum(last_price, 1)


        # average
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        hl = hl.unsqueeze(-2).repeat(1, len, 1)
        nh = self.w_1(torch.cat([pos_emb, seq_h], -1))
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs) + self.glu3(hl))
        beta = self.w_2(nh)
        beta = beta * mask
        interest_pre = torch.sum(beta * seq_h, 1)

        # multi-task
        m_p_i = torch.tanh(self.merge_w1(interest_pre) + self.merge_w2(price_pre))
        g_p = torch.sigmoid(self.merge_w3(price_pre) + self.merge_w4(m_p_i))
        g_i = torch.sigmoid(self.merge_w5(interest_pre) + self.merge_w6(m_p_i))

        p_pre = g_p * price_pre + (1 - g_p) * interest_pre
        i_pre = g_i * interest_pre + (1 - g_i) * price_pre

        return i_pre, p_pre
    def transpose_for_scores(self, x, attention_head_size):
        # INPUT:  x'shape = [bs, seqlen, hid_size]
        new_x_shape = x.size()[:-1] + (self.num_heads, attention_head_size)  # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)

    def forward(self, session_item, price_seqs, session_len, reversed_sess_item, mask):
        # session_item 是一个batch里的所有session [[23,34,0,0],[1,3,4,0]]
        item_embeddings_hg, price_embeddings_hg = self.HyperGraph(self.adjacency, self.adjacency_pp, self.adjacency_cc, self.adjacency_bb, self.adjacency_vp, self.adjacency_vc, self.adjacency_vb, self.adjacency_pv, self.adjacency_pc, self.adjacency_pb, self.adjacency_cv, self.adjacency_cp, self.adjacency_cb, self.adjacency_bv, self.adjacency_bp, self.adjacency_bc, self.embedding.weight, self.price_embedding.weight, self.category_embedding.weight, self.brand_embedding.weight) #经过三次GCN迭代的所有item embeddings
        sess_emb_hgnn, sess_pri_hgnn = self.generate_sess_emb(item_embeddings_hg, price_embeddings_hg, session_item, price_seqs, session_len, reversed_sess_item, mask) #batch内session embeddings
        v_table = self.adjacency_vp.row
        temp, idx = torch.sort(torch.tensor(v_table), dim=0, descending=False)
        vp_idx = self.adjacency_vp.col[idx]
        item_pri_l = price_embeddings_hg[vp_idx]

        return item_embeddings_hg, price_embeddings_hg, sess_emb_hgnn, sess_pri_hgnn, item_pri_l, vp_idx


def predict(model, i, data):
    tar, session_len, session_item, reversed_sess_item, mask, price_seqs = data.get_slice(i) # 得到一个batch里的数据
    # A_hat, D_hat = data.get_overlap(session_item)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    price_seqs = trans_to_cuda(torch.Tensor(price_seqs).long())
    # A_hat = trans_to_cuda(torch.Tensor(A_hat))
    # D_hat = trans_to_cuda(torch.Tensor(D_hat))
    tar_interest = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    item_emb_hg, price_emb_hg, sess_emb_hgnn, sess_pri_hgnn, item_pri_l, vp_idx = model(session_item, price_seqs, session_len, reversed_sess_item, mask)
    scores_interest = torch.mm(sess_emb_hgnn, torch.transpose(item_emb_hg, 1, 0))
    scores_price = torch.mm(sess_pri_hgnn, torch.transpose(price_emb_hg, 1, 0))
    tar_price = trans_to_cuda(torch.Tensor(vp_idx[tar]).long())
    return tar_interest, tar_price, scores_interest, scores_price


def infer(model, i, data):
    tar, session_len, session_item, reversed_sess_item, mask, price_seqs = data.get_slice(i) # 得到一个batch里的数据
    # A_hat, D_hat = data.get_overlap(session_item)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    price_seqs = trans_to_cuda(torch.Tensor(price_seqs).long())
    # A_hat = trans_to_cuda(torch.Tensor(A_hat))
    # D_hat = trans_to_cuda(torch.Tensor(D_hat))
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    item_emb_hg, price_emb_hg, sess_emb_hgnn, sess_pri_hgnn, item_pri_l, vp_idx = model(session_item, price_seqs, session_len, reversed_sess_item, mask)
    scores_interest = torch.mm(sess_emb_hgnn, torch.transpose(item_emb_hg, 1, 0))
    scores_price = torch.mm(sess_pri_hgnn, torch.transpose(item_pri_l, 1, 0))
    # scores = (1-model.beta) * torch.softmax(scores_interest, 1) + model.beta * torch.softmax(scores_price, 1)
    scores = torch.softmax(scores_interest, 1) + torch.softmax(scores_price, 1)
    return tar, scores

def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size) #将session随机打乱，每x个一组（#session/batch_size)
    for i in slices:
        model.zero_grad()
        tar_interest, tar_price, scores_interest, scores_price = predict(model, i, train_data)
        loss_interest = model.loss_function(scores_interest + 1e-8, tar_interest)
        loss_price = model.loss_function(scores_price + 1e-8, tar_price)
        loss = loss_interest + loss_price
        loss.backward()
        #        print(loss.item())
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [1, 5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
        metrics['ndcg%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        tar, scores = infer(model, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        index = np.argsort(-scores, 1)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                    metrics['ndcg%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
                    metrics['ndcg%d' % K].append(1 / (np.log2(np.where(prediction == target)[0][0] + 2)))
    return metrics, total_loss


