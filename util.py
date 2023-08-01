import numpy as np
from scipy.sparse import csr_matrix
from operator import itemgetter

def data_masks(all_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j]) #统计session中不同item，去重，并按照item_id排序
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i]-1)
            data.append(1)
    # indptr:session长度累加和; indices:item_id 减1, 由每个session内item组成; data:item在session内的权重，全部为1.
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))
    # 10000 * 6558 #sessions * #items H in paper 稀疏矩阵存储
    return matrix

def data_easy_masks(data_l, n_row, n_col):
    data, indices, indptr  = data_l[0], data_l[1], data_l[2]

    matrix = csr_matrix((data, indices, indptr), shape=(n_row, n_col))
    # 10000 * 6558 #sessions * #items H in paper 稀疏矩阵存储
    return matrix

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

class Data():
    def __init__(self, data, shuffle=False, n_node=None, n_price=None, n_category=None, n_brand=None):
        # data_formate: sessions, price_seq, matrix_session_item, matrix_session_price, matrix_session_category, matrix_session_brand, matrix_pv, matrix_pb, matrix_pc, matrix_bv, matrix_bc, matrix_cv, labels
        # session length
        self.raw = np.asarray(data[0])  # sessions, item_seq
        # self.raw = self.raw[:-1]
        self.price_raw = np.asarray(data[1])  # price_seq
        # self.price_raw = self.price_raw[:-1]

        H_T = data_easy_masks(data[2], len(data[0]), n_node)  # 10000 * 6558 #sessions * #items H_T in paper 稀疏矩阵存储
        BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
        BH_T = BH_T.T
        H = H_T.T
        DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
        DH = DH.T
        DHBH_T = np.dot(DH, BH_T)  # adjacent matrix of item to item (#item * #item)

        H_p_T = data_easy_masks(data[3], len(data[0]), n_price)  # #sessions * #price
        BH_p_T = H_p_T.T.multiply(1.0 / H_p_T.sum(axis=1).reshape(1, -1))
        BH_p_T = BH_p_T.T
        H_p = H_p_T.T
        DH_p = H_p.T.multiply(1.0 / H_p.sum(axis=1).reshape(1, -1))
        DH_p = DH_p.T
        DHBH_p_T = np.dot(DH_p, BH_p_T)  # adjacent matrix of price to price (#price * #price)

        H_c_T = data_easy_masks(data[4], len(data[0]), n_category)  # #sessions * #category
        BH_c_T = H_c_T.T.multiply(1.0 / H_c_T.sum(axis=1).reshape(1, -1))
        BH_c_T = BH_c_T.T
        H_c = H_c_T.T
        DH_c = H_c.T.multiply(1.0 / H_c.sum(axis=1).reshape(1, -1))
        DH_c = DH_c.T
        DHBH_c_T = np.dot(DH_c, BH_c_T)  # adjacent matrix of price to price (#price * #price)

        H_b_T = data_easy_masks(data[5], len(data[0]), n_brand)  # #sessions * #brand
        BH_b_T = H_b_T.T.multiply(1.0 / H_b_T.sum(axis=1).reshape(1, -1))
        BH_b_T = BH_b_T.T
        H_b = H_b_T.T
        DH_b = H_b.T.multiply(1.0 / H_b.sum(axis=1).reshape(1, -1))
        DH_b = DH_b.T
        DHBH_b_T = np.dot(DH_b, BH_b_T)  # adjacent matrix of price to price (#price * #price)

        H_pv = data_easy_masks(data[6], n_price, n_node)  # 稀疏矩阵存储
        BH_pv = H_pv  # adjacent matrix of price to item (#price * #item)
        BH_vp = H_pv.T

        H_pb = data_easy_masks(data[7], n_price, n_brand)  # 稀疏矩阵存储
        BH_pb = H_pb  # adjacent matrix of price to item (#price * #item)
        BH_bp = H_pb.T

        H_pc = data_easy_masks(data[8], n_price, n_category)  # 稀疏矩阵存储
        BH_pc = H_pc
        BH_cp = H_pc.T

        H_bv = data_easy_masks(data[9], n_brand, n_node)  # 稀疏矩阵存储
        BH_bv = H_bv  # adjacent matrix of price to item (#price * #item)
        BH_vb = H_bv.T

        H_bc = data_easy_masks(data[10], n_brand, n_category)  # 稀疏矩阵存储
        BH_bc = H_bc  # adjacent matrix of price to item (#price * #item)
        BH_cb = H_bc.T

        H_cv = data_easy_masks(data[11], n_category, n_node)  # 稀疏矩阵存储
        BH_cv = H_cv

        BH_vc = H_cv.T

        self.adjacency = DHBH_T.tocoo()
        self.adjacency_pp = DHBH_p_T.tocoo()
        self.adjacency_cc = DHBH_c_T.tocoo()
        self.adjacency_bb = DHBH_b_T.tocoo()

        self.adjacency_pv = BH_pv.tocoo()
        self.adjacency_pc = BH_pc.tocoo()
        self.adjacency_pb = BH_pb.tocoo()

        self.adjacency_vp = BH_vp.tocoo()
        self.adjacency_vc = BH_vc.tocoo()
        self.adjacency_vb = BH_vb.tocoo()

        self.adjacency_cp = BH_cp.tocoo()
        self.adjacency_cv = BH_cv.tocoo()
        self.adjacency_cb = BH_cb.tocoo()

        self.adjacency_bv = BH_bv.tocoo()
        self.adjacency_bc = BH_bc.tocoo()
        self.adjacency_bp = BH_bp.tocoo()

        self.n_node = n_node
        self.n_price = n_price
        self.n_category = n_category
        self.n_brand = n_brand
        self.targets = np.asarray(data[12])
        self.length = len(self.raw)
        self.shuffle = shuffle

    def get_overlap(self, sessions):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            for j in range(i+1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0]*len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0/degree)
        return matrix, degree

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            # 打乱session item_seq&price_seq的顺序
            self.raw = self.raw[shuffled_arg]
            self.price_raw = self.price_raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        # session length, the number of session < batchsize
        if self.length < batch_size:
            iterative = int(batch_size / self.length)
            slices = []
            slices += list(np.arange(0, self.length)) * iterative
            if batch_size % self.length != 0:
                slices += list(np.arange(0, batch_size - iterative * self.length))
            slices = [slices]
        else:
            n_batch = int(self.length / batch_size)
            if self.length % batch_size != 0:
                n_batch += 1
            slices = np.split(np.arange(n_batch * batch_size), n_batch)
            slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index):
        items, num_node, price_seqs = [], [], []
        inp = self.raw[index]
        inp_price = self.price_raw[index]
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        session_len = []
        reversed_sess_item = []
        mask = []
        for session, price in zip(inp,inp_price):
            # for session length experiments
            session = list(session)
            price = list(price)
            nonzero_elems = np.nonzero(session)[0]
            session_len.append([len(nonzero_elems)])
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            price_seqs.append(price + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1]*len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])


        return self.targets[index]-1, session_len,items, reversed_sess_item, mask, price_seqs


