import numpy as np
import pandas as pd
import joblib
import os
import sys
import time

from itertools import combinations, product
from numpy.random import seed, shuffle
from scipy.linalg import orth
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, dia_matrix, load_npz, save_npz
from scipy.sparse.linalg import eigsh
from scipy.special import comb

U_MAX = 500
PHI_UB, PHI_LB = 100, 0


#
# Preliminary preparation
#


def preliminary_preparation(alpha, l, P, parameters_only=False, time_it=False):

    if 1 <= P <= l:
        pass
    else:
        print('"P" not in the right range.')
        sys.exit()

    # Set start time
    start_time = time.perf_counter()

    # Set global parameters
    set_global_parameters(alpha, l, P, time_it)

    # Prepare D kernel basis
    if not parameters_only:
        prepare_D_kernel_basis()

    # Report execution time
    if time_it:
        print('Execution time = %.2f sec' % (time.perf_counter() - start_time))


def set_global_parameters(alpha0, l0, P0, time_it0):

    # Set global parameters for later use
    global alpha, l, P, time_it, G, s, sequences, seq_to_pos_converter, D_combs

    print('Setting global parameters ...')
    start_time = time.perf_counter()

    alpha = alpha0
    l = l0
    P = P0
    time_it = time_it0

    G = alpha**l
    s = comb(l,P) * (alpha-1)**P * alpha**(l-P)
    sequences = list(product(range(alpha), repeat=l))
    seq_to_pos_converter = np.flip(alpha**np.array(range(l)))
    D_combs = list(combinations(range(l), r=P))

    if time_it:
        print('%.2f sec' % (time.perf_counter() - start_time))
        
        
def prepare_D_kernel_basis(path='sparse_matrix/D_kernel_basis/'):

    # Set global parameters for later use
    global num_indicators, indicators_sparse, D_kernel_dim, D_kernel_basis_sparse
    
    # Get list of current sparse matrices
    spm_list = os.listdir(path)

    # If the matrix desired has been made already, load it. Otherwise, construct and save it
    file_name1 = 'indicators_alpha'+str(alpha)+'_l'+str(l)+'_P'+str(P)+'.npz'
    file_name2 = 'D_kernel_basis_alpha'+str(alpha)+'_l'+str(l)+'_P'+str(P)+'.npz'

    if (file_name1 in spm_list) and (file_name2 in spm_list):

        print('Loading D kernel basis ...')
        start_time = time.perf_counter()
        indicators_sparse = load_npz(path+file_name1)
        D_kernel_basis_sparse = load_npz(path+file_name2)
        if time_it:
            print('%.2f sec' % (time.perf_counter() - start_time))

        num_indicators = indicators_sparse.shape[1]
        D_kernel_dim = D_kernel_basis_sparse.shape[1]

    else:

        print('Constructing D kernel basis ...')
        start_time = time.perf_counter()
        indicators_sparse, D_kernel_basis_sparse = construct_D_kernel_basis()
        if time_it:
            print('%.2f sec' % (time.perf_counter() - start_time))
            
        num_indicators = indicators_sparse.shape[1]
        D_kernel_dim = D_kernel_basis_sparse.shape[1]

        save_npz(path+file_name1, indicators_sparse)
        save_npz(path+file_name2, D_kernel_basis_sparse)
    
    
def construct_D_kernel_basis():

    # Construct indicator functions first
    indicators = []
    for p in range(P):        
        site_groups = list(combinations(range(l), r=p))
        base_groups = list(product(range(alpha), repeat=p))        
        for site_group in site_groups:
            for base_group in base_groups:        
                u = np.ones([alpha]*l)
                for k, site in enumerate(site_group):
                    v = np.zeros([alpha]*l)
                    v[base_group[k],...] = 1
                    v = np.swapaxes(v, 0, site)
                    u *= v            
                indicators.append(u.ravel())
    
    indicators = np.array(indicators).T
    
    # Construct an orthonormal basis for the kernel of D
    D_kernel_basis = orth(indicators)
    
    # Save indicators and D_kernel_basis as a sparse matrix
    indicators_sparse = csr_matrix(indicators)
    D_kernel_basis_sparse = csr_matrix(D_kernel_basis)
    
    # Return
    return indicators_sparse, D_kernel_basis_sparse


#
# Data importation
#


def import_data(path, coding_dict, ignore_sites=None):

    # Read in processed data
    df = pd.read_csv(path, sep='\s+', names=['sequence', 'count'], dtype=str)

    # Get flags for the sites of interest
    if ignore_sites is not None:
        flags = np.full(l+len(ignore_sites), True)
        flags[ignore_sites] = False

    # Obtain count data
    Ns = np.zeros(G)
    for i in range(len(df)):
        sequence, count = df.loc[i, ['sequence','count']]
        try:  # sequences with letters not included in coding_dict will be ignored
            seq = [coding_dict[letter] for letter in sequence]
            if ignore_sites is not None:
                seq = np.array(seq)[flags]
            pos = sequence_to_position(seq)
            Ns[pos] = int(count)
        except:
            pass

    # Normalize count data
    N = np.sum(Ns)
    R = Ns / N

    # Save N and R
    data_dict = {'N': int(N), 'R': R}

    # Return
    return data_dict
    

#
# MAP estimation
#
    
    
def estimate_MAP_solution(a, data_dict, phi_initial=None, method='L-BFGS-B', options=None, scale_by=1):

    # Set start time
    start_time = time.perf_counter()

    # Get N and R
    N, R = data_dict['N'], data_dict['R']

    # Do scaling
    a /= scale_by
    N /= scale_by

    # Find the MAP estimate of phi
    if a == 0:

        with np.errstate(divide='ignore'):
            phi_a = -np.log(R)

    elif 0 < a < np.inf:
        
        # Set initial guess of phi
        if phi_initial is None:
            Q_initial = np.ones(G) / G
            phi_initial = -np.log(Q_initial)

        res = minimize(fun=S, jac=grad_S, args=(a,N,R), x0=phi_initial, method=method, options=options)
        if not res.success:
            print(res.message)
            print()
        phi_a = res.x

    elif a == np.inf:

        b_initial = np.ones(D_kernel_dim)
        res = minimize(fun=S_inf, jac=grad_S_inf, args=(N,R), x0=b_initial, method=method, options=options)
        if not res.success:
            print(res.message)
            print()
        b_a = res.x
        phi_a = D_kernel_basis_sparse.dot(b_a)
        
    else:

        print('"a" not in the right range.')
        sys.exit()

    # Undo scaling
    a *= scale_by
    N *= scale_by

    # Report execution time
    if time_it:
        print('Execution time = %.2f sec' % (time.perf_counter() - start_time))

    # Return
    return phi_a


def trace_MAP_curve(data_dict, resolution=0.1, num_a=20, fac_max=1, fac_min=1e-3, options=None, scale_by=1):

    # Set start time
    start_time = time.perf_counter()

    # Create a container dataframe
    df_map = pd.DataFrame(columns=['a', 'phi'])

    # Compute a = inf end
    print('Computing a = inf ...')
    a_inf = np.inf
    phi_inf = estimate_MAP_solution(a_inf, data_dict, phi_initial=None, options=options, scale_by=scale_by)
    df_map = df_map.append({'a': a_inf, 'phi': phi_inf}, ignore_index=True)

    # Find a_max that is finite and close enough to a = inf
    a_max = s * fac_max
    print('Computing a_max = %f ...' % a_max)
    phi_max = estimate_MAP_solution(a_max, data_dict, phi_initial=phi_inf, options=options, scale_by=scale_by)
    print('... D_geo(Q_max, Q_inf) = %f' % D_geo(phi_max, phi_inf))
    while D_geo(phi_max, phi_inf) > resolution:
        a_max *= 10
        print('Computing a_max = %f ...' % a_max)
        phi_max = estimate_MAP_solution(a_max, data_dict, phi_initial=phi_inf, options=options, scale_by=scale_by)
        print('... D_geo(Q_max, Q_inf) = %f' % D_geo(phi_max, phi_inf))
    df_map = df_map.append({'a': a_max, 'phi': phi_max}, ignore_index=True)

    # Compute a = 0 end
    print()
    print('Computing a = 0 ...')
    a_0 = 0
    phi_0 = estimate_MAP_solution(a_0, data_dict, phi_initial=None, options=options, scale_by=scale_by)
    df_map = df_map.append({'a': a_0, 'phi': phi_0}, ignore_index=True)

    # Find a_min that is finite and close enough to a = 0
    a_min = s * fac_min
    print('Computing a_min = %f ...' % a_min)
    phi_min = estimate_MAP_solution(a_min, data_dict, phi_initial=phi_inf, options=options, scale_by=scale_by)
    print('... D_geo(Q_min, Q_0) = %f' % D_geo(phi_min, phi_0))
    while D_geo(phi_min, phi_0) > resolution:
        a_min /= 10
        print('Computing a_min = %f ...' % a_min)
        phi_min = estimate_MAP_solution(a_min, data_dict, phi_initial=phi_inf, options=options, scale_by=scale_by)
        print('... D_geo(Q_min, Q_0) = %f' % D_geo(phi_min, phi_0))
    df_map = df_map.append({'a': a_min, 'phi': phi_min}, ignore_index=True)

    # Compute 0 < a < inf
    if num_a is None:

        # Gross-partition the MAP curve
        print()
        print('Gross-partitioning the MAP curve ...')
        aa = np.geomspace(a_min, a_max, 10)
        for i in range(len(aa)-2, 0, -1):
            a = aa[i]
            print('Computing a = %f ...' % a)
            phi_a = estimate_MAP_solution(a, data_dict, phi_initial=phi_inf, options=options, scale_by=scale_by)
            df_map = df_map.append({'a': a, 'phi': phi_a}, ignore_index=True)

        # Fine-partition the MAP curve to achieve desired resolution
        print()
        print('Fine-partitioning the MAP curve ...')
        flag = True
        while flag:
            df_map = df_map.sort_values(by='a')
            aa, phis = df_map['a'].values, df_map['phi'].values
            flag = False
            for i in range(len(df_map)-1):
                a_i, a_j = aa[i], aa[i+1]
                phi_i, phi_j = phis[i], phis[i+1]
                if D_geo(phi_i, phi_j) > resolution:
                    a = np.geomspace(a_i, a_j, 3)[1]
                    print('Computing a = %f ...' % a)
                    phi_a = estimate_MAP_solution(a, data_dict, phi_initial=phi_inf, options=options, scale_by=scale_by)
                    df_map = df_map.append({'a': a, 'phi': phi_a}, ignore_index=True)
                    flag = True

    else:

        # Partition the MAP curve into num_a points
        print()
        print('Partitioning the MAP curve into %d points ...' % num_a)
        aa = np.geomspace(a_min, a_max, num_a)
        for i in range(len(aa)-2, 0, -1):
            a = aa[i]
            print('Computing a_%d = %f ...' % (i, a))
            phi_a = estimate_MAP_solution(a, data_dict, phi_initial=phi_inf, options=options, scale_by=scale_by)
            df_map = df_map.append({'a': a, 'phi': phi_a}, ignore_index=True)
        df_map = df_map.sort_values(by='a')

    df_map = df_map.sort_values(by='a')
    df_map = df_map.reset_index(drop=True)

    # Report total execution time
    if time_it:
        print('Total execution time = %.2f sec' % (time.perf_counter() - start_time))

    # Return
    return df_map


#
# Cross validation
#


def compute_log_likelihoods(data_dict, df_map, cv_fold=5, random_seed=None, options=None, scale_by=1):

    # Set start time
    start_time = time.perf_counter()

    # Generate training sets and validation sets
    df_train_data, df_valid_data = split_data(data_dict, cv_fold, random_seed)

    # Compute log_Ls averaged over k folds
    log_Lss = np.zeros([cv_fold,len(df_map)])

    for k in range(cv_fold):

        print('Doing cross validation fold # %d ...' % k)

        N_train, R_train = df_train_data.loc[k, ['N','R']]
        N_valid, R_valid = df_valid_data.loc[k, ['N','R']]

        data_dict_train = {'N': N_train, 'R': R_train}
        Ns_valid = N_valid * R_valid

        # For each a, compute Q with training set and compute log_L with validation set
        for i in range(len(df_map)):
            print('i = %d' % i)
            a, phi_a = df_map.loc[i, ['a','phi']]
            phi = estimate_MAP_solution(a, data_dict_train, phi_initial=phi_a, options=options, scale_by=scale_by)
            Q = np.exp(-phi) / np.sum(np.exp(-phi))
            if a == 0:
                Q_flags = (Q == 0)
                if np.sum(Q_flags) > 0:
                    log_L = -np.inf
                else:
                    log_L = np.sum(Ns_valid * np.log(Q))
            else:
                log_L = np.sum(Ns_valid * np.log(Q))
            log_Lss[k,i] = log_L

    log_Ls = log_Lss.mean(axis=0)

    # Save log_Ls
    df_map['log_L'] = log_Ls

    # Report execution time
    if time_it:
        print('Execution time = %.2f sec' % (time.perf_counter() - start_time))

    # Return
    return df_map


def split_data(data_dict, cv_fold, random_seed=None):

    # Get N and R
    N, R = data_dict['N'], data_dict['R']

    # Generate raw data
    raw_data = generate_raw_data(data_dict, random_seed)

    # Reshape raw data into an array of k=cv_fold rows
    remainder = N % cv_fold
    row_len = int((N-remainder) / cv_fold)
    raw_data_array = np.reshape(raw_data[:N-remainder], (cv_fold, row_len))

    # If some raw data are left, create a dictionary to map each raw datum left to one k
    if remainder != 0:
        raw_data_left = raw_data[-remainder:]
        left_dict = {}
        for k, raw_datum_left in enumerate(raw_data_left):
            left_dict[k] = raw_datum_left
        left_dict_keys = list(left_dict.keys())

    # Split raw data into training sets and validation sets
    df_train_data, df_valid_data = pd.DataFrame(columns=['N', 'R']), pd.DataFrame(columns=['N', 'R'])

    for k in range(cv_fold):

        # Get training data
        ids = list(range(cv_fold))
        ids.remove(k)
        train_data = raw_data_array[ids,:].reshape(-1)
        if remainder != 0:
            for id in ids:
                if id in left_dict_keys:
                    train_data = np.append(train_data, left_dict[id])
        values, counts = np.unique(train_data, return_counts=True)
        Ns_train = np.zeros(G)
        Ns_train[values] = counts
        N_train = np.sum(counts)
        R_train = Ns_train / N_train
        df_train_data = df_train_data.append({'N': N_train, 'R': R_train}, ignore_index=True)

        # Get validation data
        valid_data = raw_data_array[k,:]
        if remainder != 0:
            if k in left_dict_keys:
                valid_data = np.append(valid_data, left_dict[k])
        values, counts = np.unique(valid_data, return_counts=True)
        Ns_valid = np.zeros(G)
        Ns_valid[values] = counts
        N_valid = np.sum(counts)
        R_valid = Ns_valid / N_valid
        df_valid_data = df_valid_data.append({'N': N_valid, 'R': R_valid}, ignore_index=True)

    # Return
    return df_train_data, df_valid_data


def generate_raw_data(data_dict, random_seed=None):

    # Set random seed
    seed(random_seed)

    # Get N and R
    N, R = data_dict['N'], data_dict['R']

    # Generate raw data
    Ns = N * R
    raw_data = []
    for i in range(G):
        raw_data.extend([i]*int(round(Ns[i])))
    raw_data = np.array(raw_data)

    # Make sure the amount of raw data is correct
    if len(raw_data) != N:
        print('"raw_data" not correctly generated.')
        sys.exit()

    # Shuffle raw data
    shuffle(raw_data)

    # Return
    return raw_data


#
# Analysis tools: computing p-th order associations
#


def compute_rms_log_p_association(phi, p):
    phi = np.reshape(phi, [alpha]*l)
    D_p_combs = list(combinations(range(l), r=p))
    s_p = comb(l,p) * (alpha-1)**p * alpha**(l-p)
    if np.any(phi == np.inf):
        rms_log_Ap = np.inf
    else:
        phiDphi = 0
        for D_p_comb in D_p_combs:
            phi_tmp = np.copy(phi)
            for i in D_p_comb:
                phi_tmp = np.diff(phi_tmp, axis=i)
            phiDphi += np.sum(phi_tmp**2)
        rms_log_Ap = np.sqrt(1/s_p * phiDphi)
    return rms_log_Ap


def compute_log_p_associations(phi, sites_dict, condition_dict={}, coding_dict=None):

    # If coding dictionary is provided, convert letters to codes
    if coding_dict is not None:
        if None not in sites_dict.values():
            for key in sites_dict.keys():
                value = [coding_dict[letter] for letter in sites_dict[key]]
                sites_dict[key] = value
        for key in condition_dict.keys():
            value = [coding_dict[letter] for letter in condition_dict[key]]
            condition_dict[key] = value
    
    # Get sites and mutations
    sites = list(sites_dict.keys())
    mutations = list(sites_dict.values())

    # Generate bases
    bases = list(range(alpha))

    # Get background sites
    bg_sites = list(set(range(l)) - set(sites))

    # Get allowable bases for each background site
    bg_sites_bases = []
    for bg_site in bg_sites:
        if bg_site in condition_dict.keys():
            bg_sites_bases.append(condition_dict[bg_site])
        else:
            bg_sites_bases.append(bases)

    # Generate background sequences
    bg_seqs = list(product(*bg_sites_bases))

    # Generate all possible units that can be formed by mutations at sites
    if None not in mutations:
        for mutation in mutations:
            if abs(mutation[1]-mutation[0]) != 1:
                print('Mutation(s) not legitimate.')
                sys.exit()
        units = [list(product(*mutations))]
    else:
        base_pairs = []
        for i in range(alpha-1):
            base_pairs.append([i,i+1])
        base_pair_products = list(product(base_pairs, repeat=len(sites)))
        units = []
        for base_pair_product in base_pair_products:
            units.append(list(product(*base_pair_product)))
    
    # For each background sequence, compute log_Ap on all units formed by mutations at sites
    log_Aps, associated_seqs = [], []
    for bg_seq in bg_seqs:
        for unit in units:
            unit_phis, unit_seqs = [], []
            for k in range(len(unit)):
                unit_vertex_k_seq = np.full(l, -1, dtype=int)
                unit_vertex_k_seq[bg_sites] = bg_seq
                unit_vertex_k_seq[sites] = unit[k]
                unit_vertex_k_pos = sequence_to_position(unit_vertex_k_seq)
                unit_phis.append(phi[unit_vertex_k_pos])
                unit_seqs.append(list(unit_vertex_k_seq))
            unit_phis = np.reshape(unit_phis, [2]*len(sites))
            for i in range(len(sites)):
                unit_phis = np.diff(unit_phis, axis=i)
            log_Aps.append(-unit_phis.ravel()[0])
            associated_seqs.append(unit_seqs)
            
    # If coding dictionary is provided, convert codes to letters
    if coding_dict is not None:
        rev_coding_dict = dict(map(reversed, coding_dict.items()))
        TMP = []
        for seqs in associated_seqs:
            tmp = []
            for seq in seqs:
                tmp.append(''.join([rev_coding_dict[code] for code in seq]))
            TMP.append(tmp)
        associated_seqs = TMP

    # Save log_Aps and associated sequences in a dataframe
    df_log_Aps = pd.DataFrame()
    df_log_Aps['log_Ap'], df_log_Aps['associated_seqs'] = log_Aps, associated_seqs
    df_log_Aps = df_log_Aps.sort_values(by='log_Ap', ascending=False).reset_index(drop=True)

    # Return
    return df_log_Aps


#
# Analysis tools: making visualization
#


def make_visualization(Q, markov_chain, K=20, tol=1e-9, reuse_Ac=False, path='sparse_matrix/Ac/'):

    # Set start time
    Start_time = time.perf_counter()

    # If reuse existing A and c, load them. Otherwise, construct A and c from scratch and save them
    if reuse_Ac:

        print('Loading A and c ...')
        start_time = time.perf_counter()
        A_sparse = load_npz(path+'A.npz')
        c = joblib.load(path+'c.pkl')
        if time_it:
            print('%.2f sec' % (time.perf_counter() - start_time))

    else:

        print('Constructing A and c ...')
        start_time = time.perf_counter()
        A_sparse, c = construct_Ac(Q, markov_chain)
        if time_it:
            print('%.2f sec' % (time.perf_counter() - start_time))

        save_npz(path+'A.npz', A_sparse)
        joblib.dump(c, path+'c.pkl')

    # Compute the dominant eigenvalues and eigenvectors of A
    print('Computing dominant eigenvalues and eigenvectors of A ...')
    start_time = time.perf_counter()
    eig_vals_tilt, eig_vecs_tilt = eigsh(A_sparse, K, which='LM', tol=tol)
    if time_it:
        print('%.2f sec' % (time.perf_counter() - start_time))

    # Check accuracy of the eigenvalues and eigenvectors of A
    df_check = pd.DataFrame(columns=['eigenvalue', 'colinearity', 'max_difference'])
    for k in range(K):
        lda, u = eig_vals_tilt[k], eig_vecs_tilt[:,k]
        Au = A_sparse.dot(u)
        max_diff = abs(Au-lda*u).max()
        Au /= np.sqrt(np.sum(Au**2))
        colin = np.sum(Au*u)
        df_check = df_check.append({'eigenvalue': lda, 'colinearity': colin, 'max_difference': max_diff}, ignore_index=True)
    df_check = df_check.sort_values(by='eigenvalue', ascending=False).reset_index(drop=True)

    # Obtain the eigenvalues and eigenvectors of T, and use them to construct visualization coordinates
    Diag_Q_inv_sparse = dia_matrix((1/np.sqrt(Q), np.array([0])), shape=(G,G))
    df_visual = pd.DataFrame(columns=['eigenvalue', 'coordinate'])
    for k in range(K):
        lda, u = eig_vals_tilt[k], eig_vecs_tilt[:,k]
        if lda < 1:
            eig_val = c * (lda - 1)
            eig_vec = Diag_Q_inv_sparse.dot(u)
            coordinate = eig_vec / np.sqrt(-eig_val)
            df_visual = df_visual.append({'eigenvalue': eig_val, 'coordinate': coordinate}, ignore_index=True)
        else:
            df_visual = df_visual.append({'eigenvalue': 0, 'coordinate': np.full(G,np.nan)}, ignore_index=True)
    df_visual = df_visual.sort_values(by='eigenvalue', ascending=False).reset_index(drop=True)

    # Report execution time
    if time_it:
        print('Execution time = %.2f sec' % (time.perf_counter() - Start_time))

    # Return
    return df_visual, df_check


def construct_Ac(Q, markov_chain):

    # Choose a model for the reversible Markov chain
    if markov_chain == 'evolutionary':
        T_ij = T_evolutionary
    elif markov_chain == 'Metropolis':
        T_ij = T_Metropolis
    elif markov_chain == 'power_law':
        T_ij = T_power_law
    else:
        print('markov_chain "model" not recognized.')
        sys.exit()

    # Construct rate matrix T
    row_ids, col_ids, values = [], [], []
    for i in range(G):
        tmp = []
        for site in range(l):
            for shift in [+1, -1]:
                seq_i = np.array(sequences[i])
                seq_i[site] += shift
                if seq_i[site] in range(alpha):
                    j = sequence_to_position(seq_i)
                    value = T_ij(Q[i], Q[j])
                    row_ids.append(i)
                    col_ids.append(j)
                    values.append(value)
                    tmp.append(value)
        row_ids.append(i)
        col_ids.append(i)
        values.append(-np.sum(tmp))

    # Save T as a sparse matrix
    T_sparse = csr_matrix((values, (row_ids, col_ids)), shape=(G,G))

    # Construct a symmetric matrix T_tilt from T
    Diag_Q_sparse = dia_matrix((np.sqrt(Q), np.array([0])), shape=(G,G))
    Diag_Q_inv_sparse = dia_matrix((1/np.sqrt(Q), np.array([0])), shape=(G,G))
    T_tilt_sparse = Diag_Q_sparse.dot(T_sparse * Diag_Q_inv_sparse)

    # Choose the value of c
    c = 0
    for i in range(G):
        sum_i = abs(T_tilt_sparse[i,i])
        for site in range(l):
            for shift in [+1, -1]:
                seq_i = np.array(sequences[i])
                seq_i[site] += shift
                if seq_i[site] in range(alpha):
                    j = sequence_to_position(seq_i)
                    sum_i += abs(T_tilt_sparse[i,j])
        c = max(c, sum_i)

    # Construct A and save it as a sparse matrix
    I_sparse = dia_matrix((np.ones(G), np.array([0])), shape=(G,G))
    A_sparse = I_sparse + 1/c * T_tilt_sparse

    # Return
    return A_sparse, c


def T_evolutionary(Q_i, Q_j, par=1):
    if Q_i == Q_j:
        return 1
    else:
        return par * (np.log(Q_j)-np.log(Q_i)) / (1 - np.exp(-par * (np.log(Q_j)-np.log(Q_i))))


def T_Metropolis(Q_i, Q_j):
    if Q_j > Q_i:
        return 1
    else:
        return Q_j/Q_i


def T_power_law(Q_i, Q_j, par=1/2):
    return Q_j**par / Q_i**(1-par)


def get_nodes(df_visual, kx, ky, xflip=1, yflip=1):

    # Get specified visualization coordinates
    x, y = df_visual.loc[kx,'coordinate']*xflip, df_visual.loc[ky,'coordinate']*yflip

    # Save the coordinates
    df_nodes = pd.DataFrame()
    df_nodes['node'], df_nodes['x'], df_nodes['y'] = range(G), x, y

    # Return
    return df_nodes


def get_edges(df_visual, kx, ky, xflip=1, yflip=1):

    # Get specified visualization coordinates
    x, y = df_visual.loc[kx,'coordinate']*xflip, df_visual.loc[ky,'coordinate']*yflip

    # Get coordinates of all edges (i > j)
    nodes_i, nodes_j, edges = [], [], []
    for i in range(G):
        for site in range(l):
            for shift in [+1, -1]:
                seq_i = np.array(sequences[i])
                seq_i[site] += shift
                if seq_i[site] in range(alpha):
                    j = sequence_to_position(seq_i)
                    if i > j:
                        nodes_i.append(i)
                        nodes_j.append(j)
                        edges.append([(x[i],y[i]), (x[j],y[j])])

    # Save the coordinates
    df_edges = pd.DataFrame()
    df_edges['node_i'], df_edges['node_j'], df_edges['edge'] = nodes_i, nodes_j, edges

    # Return
    return df_edges


#
# Utility functions
#


def sequence_to_position(seq, coding_dict=None):
    if coding_dict is None:
        return int(np.sum(seq * seq_to_pos_converter))
    else:
        tmp = [coding_dict[letter] for letter in seq]
        return int(np.sum(tmp * seq_to_pos_converter))
    
    
def position_to_sequence(pos, coding_dict=None):
    if coding_dict is None:
        return sequences[pos]
    else:
        rev_coding_dict = dict(map(reversed, coding_dict.items()))
        tmp = sequences[pos]
        return ''.join([rev_coding_dict[code] for code in tmp])


def D_geo(phi1, phi2):
    Q1 = np.exp(-phi1) / np.sum(np.exp(-phi1))
    Q2 = np.exp(-phi2) / np.sum(np.exp(-phi2))
    x = min(np.sum(np.sqrt(Q1 * Q2)), 1)
    return 2 * np.arccos(x)


def compute_rms_log_P_association(phi):
    phi = np.reshape(phi, [alpha]*l)
    if np.any(phi == np.inf):
        rms_log_AP = np.inf
    else:
        phiDphi = 0
        for D_comb in D_combs:
            phi_tmp = np.copy(phi)
            for i in D_comb:
                phi_tmp = np.diff(phi_tmp, axis=i)
            phiDphi += np.sum(phi_tmp**2)
        rms_log_AP = np.sqrt(1/s * phiDphi)
    return rms_log_AP


def compute_marginal_probability(phi):
    Q = np.exp(-phi) / np.sum(np.exp(-phi))
    Q_ind = indicators_sparse.T.dot(Q)
    df_mprobs = pd.DataFrame(columns=['sites', 'bases', 'probability'])
    c = 0
    for p in range(P):
        site_groups = list(combinations(range(l), r=p))
        base_groups = list(product(range(alpha), repeat=p))
        for site_group in site_groups:
            for base_group in base_groups:
                df_mprobs = df_mprobs.append({'sites': site_group, 'bases': base_group, 'probability': Q_ind[c]},
                                             ignore_index=True)
                c += 1
    return df_mprobs


#
# Basic functions
#


def safe_exp(v):
    u = v.copy()
    u[u > U_MAX] = U_MAX
    return np.exp(u)


def S(phi, a, N, R):
    phi = np.reshape(phi, [alpha]*l)
    S1 = 0
    for D_comb in D_combs:
        phi_tmp = np.copy(phi)
        for i in D_comb:
            phi_tmp = np.diff(phi_tmp, axis=i)
        S1 += np.sum(phi_tmp**2)
    S1 *= a/(2*s)
    phi = phi.ravel()
    S2 = N * np.sum(R * phi)
    S3 = N * np.sum(safe_exp(-phi))
    regularizer = 0
    if np.isfinite(PHI_UB):
        flags = (phi > PHI_UB)
        if flags.sum() > 0:
            regularizer += np.sum((phi - PHI_UB)[flags]**2)
    if np.isfinite(PHI_LB):
        flags = (phi < PHI_LB)
        if flags.sum() > 0:
            regularizer += np.sum((phi - PHI_LB)[flags]**2)
    return S1 + S2 + S3 + regularizer


def grad_S(phi, a, N, R):
    phi = np.reshape(phi, [alpha]*l)
    grad_S1 = np.zeros(G)
    for D_comb in D_combs:
        phi_tmp = np.copy(phi)
        for i in D_comb:
            phi_tmp = np.diff(phi_tmp, axis=i)
        grad_tmp = np.copy(phi_tmp)
        for i in D_comb:
            shape = list(grad_tmp.shape)
            shape[i] = 1
            A = np.concatenate([grad_tmp, np.zeros(shape)], axis=i)
            B = np.concatenate([np.zeros(shape), grad_tmp], axis=i)
            grad_tmp = A - B
        grad_S1 += grad_tmp.ravel()
    grad_S1 *= (-1)**P * (a/s)
    phi = phi.ravel()
    grad_S2 = N * R
    grad_S3 = N * safe_exp(-phi)
    regularizer = np.zeros(G)
    if np.isfinite(PHI_UB):
        flags = (phi > PHI_UB)
        if flags.sum() > 0:
            regularizer[flags] += 2 * (phi - PHI_UB)[flags]
    if np.isfinite(PHI_LB):
        flags = (phi < PHI_LB)
        if flags.sum() > 0:
            regularizer[flags] += 2 * (phi - PHI_LB)[flags]
    return grad_S1 + grad_S2 - grad_S3 + regularizer


def S_inf(b, N, R):
    phi = D_kernel_basis_sparse.dot(b)
    S_inf1 = N * np.sum(R * phi)
    S_inf2 = N * np.sum(safe_exp(-phi))
    regularizer = 0
    if np.isfinite(PHI_UB):
        flags = (phi > PHI_UB)
        if flags.sum() > 0:
            regularizer += np.sum((phi - PHI_UB)[flags]**2)
    if np.isfinite(PHI_LB):
        flags = (phi < PHI_LB)
        if flags.sum() > 0:
            regularizer += np.sum((phi - PHI_LB)[flags]**2)
    return S_inf1 + S_inf2 + regularizer


def grad_S_inf(b, N, R):
    phi = D_kernel_basis_sparse.dot(b)
    grad_S_inf1 = N * R
    grad_S_inf2 = N * safe_exp(-phi)
    regularizer = np.zeros(G)
    if np.isfinite(PHI_UB):
        flags = (phi > PHI_UB)
        if flags.sum() > 0:
            regularizer[flags] += 2 * (phi - PHI_UB)[flags]
    if np.isfinite(PHI_LB):
        flags = (phi < PHI_LB)
        if flags.sum() > 0:
            regularizer[flags] += 2 * (phi - PHI_LB)[flags]
    return D_kernel_basis_sparse.T.dot(grad_S_inf1 - grad_S_inf2 + regularizer)

