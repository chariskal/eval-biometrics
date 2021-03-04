import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
import os
import pandas as pd                 # package for data analysis with fast and flexible data structures
import logging
import src.BSSR1 as BSSR1
import src.eval as ev
import xml.etree.ElementTree as ET  # package for reading xml files
from tqdm.notebook import tqdm as tqdm_notebook

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('./logs/evaluation.log')
handler.setLevel(logging.DEBUG)

f_format = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
handler.setFormatter(f_format)
logger.addHandler(handler)

def read_data():
    li_enrollees_id_filepath = './data/fing_x_fing/sets/dos/li/enrollees.xml'
    li_users_id_filepath = './data/fing_x_fing/sets/dos/li/users.xml'
    ri_enrollees_id_filepath = './data/fing_x_fing/sets/dos/ri/enrollees.xml'
    ri_users_id_filepath = './data/fing_x_fing/sets/dos/ri/users.xml'

    # set list string for scores files
    li_scores_list_str = './data/fing_x_fing/sims/dos/li/V/*.bin'
    ri_scores_list_str = './data/fing_x_fing/sims/dos/ri/V/*.bin'

    (li_fingxfing_df, _li_enrollees, _li_users, li_column_names) = BSSR1.read_BSSR1_scores_from_file(li_enrollees_id_filepath, li_users_id_filepath, li_scores_list_str)
    li_sim = BSSR1.df2sim_subsample(li_fingxfing_df, column_names = li_column_names, nr_individuals = 1000)
    li_genuine_id, li_scores = BSSR1.sim2scores(li_sim)
    del li_fingxfing_df


    (ri_fingxfing_df, _ri_enrollees, _ri_users, ri_column_names) = BSSR1.read_BSSR1_scores_from_file(ri_enrollees_id_filepath, ri_users_id_filepath, ri_scores_list_str)
    ri_sim = BSSR1.df2sim_subsample(ri_fingxfing_df, column_names = ri_column_names, nr_individuals = 1000)
    ri_genuine_id, ri_scores = BSSR1.sim2scores(ri_sim)
    del ri_fingxfing_df
    logger.info(f'BSSR1 data read!')
    return li_genuine_id, li_scores, ri_genuine_id, ri_scores


if __name__ == '__main__':
    [li_genuine_id, li_scores, ri_genuine_id, ri_scores] = read_data()
    num_subjects = 1000

    V = ev.Eval_class(num_subjects)

    sc_list = [li_scores, ri_scores]
    y_true_list = [li_genuine_id, ri_genuine_id]
    labels = ['left index', 'right index']
    V.plot_distribution(sc_list, labels)

    V.plot_ROC(y_true_list, sc_list, labels)
    [auc_li, auc_ri] = V.calc_auc(y_true_list, sc_list)
    logger.info(f'AUC for ROC (left index): {auc_li}, (right index) {auc_ri}')
    V.plot_errvth(y_true_list, sc_list, labels)
    V.plot_det(y_true_list, sc_list, labels)
    [f1_th_max, acc_th_max] = V.plot_f1_acc(y_true_list, sc_list, labels)
    logger.info(f'F1-score is maximum (left index) for thres: {f1_th_max[0]}, (right index) {f1_th_max[1]}')
    logger.info(f'Accuracy is maximum (left index) for thres: {acc_th_max[0]}, (right index) {acc_th_max[1]}')

    l = V.plot_eer(y_true_list, sc_list, labels)
    auprc_list, ap_list = V.plot_prc(y_true_list, sc_list, labels)
    logger.info(f'AUC for precision-recall (left index) is: {auprc_list[0]}, (right index) {auprc_list[1]}')
    logger.info(f'Average Precision score (left index) is: {ap_list[0]}, (right index) {ap_list[1]}')

    rank1_list = V.plot_cmc(y_true_list, sc_list, labels)
    logger.info(f'Rank-1 rate (left index) is: {rank1_list[0]}, (right index) {rank1_list[1]}')
    logger.info('Script ended succesfully!')
