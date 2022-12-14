import numpy as np
import pandas as pd
import uproot
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
from pandas import MultiIndex, Int16Dtype
import random
import matplotlib.pyplot as plt

from tools.xgboost2tmva import *

trainvar_doc = "/eos/user/x/xisu/WorkSpace/VH_AC_Analysis/Data_MC_comparision/CMSSW_12_3_0/src/MakePlot/RR_18_v2/TrainingVar/generate_sideband_var_withbdtscore_fa31d0/outputfiles"

smsig_name = ["/outvar_ZHiggs0PMToGG_M125_13TeV_JHUGenV7011_pythia8.root", 
              "/outvar_WHiggs0PMToGG_M125_13TeV_JHUGenV7011_pythia8.root"]

acsig_fa31d0_name = ["/outvar_ZHiggs0MToGG_M125_13TeV_JHUGenV7011_pythia8.root",
                     "/outvar_WHiggs0MToGG_M125_13TeV_JHUGenV7011_pythia8.root"]


Prob_hist_title = [["ZHiggs0MToGG_M125"], 
                   ["WHiggs0MToGG_M125"]]

varset_name = ["adding_met_info"]

def main():
    for channel_type in range(2):
        file_smsig = uproot.open(trainvar_doc + smsig_name[channel_type])
        file_acsig = uproot.open(trainvar_doc + acsig_fa31d0_name[channel_type])
        # check the tree names are the same in both sm and ac samples
        print(file_smsig.keys())
        if file_smsig.keys() != file_acsig.keys():
            print("tree name are differnt between background and signal, please check import files")
            continue
        
        print("Loading SM file: {}".format(smsig_name[channel_type]))
        print("Loading AC file: {}".format(acsig_fa31d0_name[channel_type]))

        tree_loc = file_smsig.keys()[0]
        print(tree_loc)
        tree_smsig = file_smsig[tree_loc]
        tree_acsig = file_acsig[tree_loc]

        # dataset = [
        #     "pho1_eta",
        #     "pho2_eta",
        #     "pho1_phi",
        #     "pho2_phi",
        #     "pho1_ptoM",
        #     "pho2_ptoM",
        #     "dipho_cosphi",
        #     "dipho_deltaeta",
        #     "met",
        #     "met_phi",
        #     "met_sumEt",
        #     "dphi_dipho_met",
        #     "pt_balance",
        #     "njet",
        #     "max_jet_pt",
        #     "min_dphi_jet_met",
        # ]

        # dataset_forxml = [
        #     ("pho1_eta", "F"),
        #     ("pho2_eta", "F"),
        #     ("pho1_phi", "F"),
        #     ("pho2_phi", "F"),
        #     ("pho1_ptoM", "F"),
        #     ("pho2_ptoM", "F"),
        #     ("dipho_cosphi", "F"),
        #     ("dipho_deltaeta", "F"),
        #     ("met", "F"),
        #     ("met_phi", "F"),
        #     ("met_sumEt", "F"),
        #     ("dphi_dipho_met", "F"),
        #     ("pt_balance", "F"),
        #     ("njet", "F"),
        #     ("max_jet_pt", "F"),
        #     ("min_dphi_jet_met", "F"),
        # ]

        dataset = [
            "pho1_eta",
            "pho2_eta",
            "pho1_phi",
            "pho2_phi",
            "pho1_ptoM",
            "pho2_ptoM",
			"pho1_R9",
            "pho2_R9",
            "pho1_sieie",
            "pho2_sieie",
            "dipho_cosphi",
            "dipho_deltaeta",
            "met",
            "met_phi",
            "met_sumEt",
            "dphi_pho1_met",
            "dphi_pho2_met",
            "pt_balance",
            "njet",
            "max_jet_pt",
            "min_dphi_jet_met",
        ]

        dataset_forxml = [
            ("pho1_eta", "F"),
            ("pho2_eta", "F"),
            ("pho1_phi", "F"),
            ("pho2_phi", "F"),
            ("pho1_ptoM", "F"),
            ("pho2_ptoM", "F"),
            ("pho1_R9", "F"),
            ("pho2_R9", "F"),
            ("pho1_sieie", "F"),
            ("pho2_sieie", "F"),
            ("dipho_cosphi", "F"),
            ("dipho_deltaeta", "F"),
            ("met", "F"),
            ("met_phi", "F"),
            ("met_sumEt", "F"),
            ("dphi_pho1_met", "F"),
            ("dphi_pho2_met", "F"),
            ("pt_balance", "F"),
            ("njet", "F"),
            ("max_jet_pt", "F"),
            ("min_dphi_jet_met", "F"),
        ]

        featvarset = [dataset]
        featvarset_forxml = [dataset_forxml]

        # prepare empty lists for storing accuracy score imformation
        # and ROC curve indormations for different featuring paramenter set
        predict_acc = []
        fpr_test_list = []
        tpr_test_list = []
        roc_auc_test_list = []

        for setentry in range(len(featvarset)):
            smsig = tree_smsig.arrays(featvarset[setentry], library="pd")
            acsig = tree_acsig.arrays(featvarset[setentry], library="pd")
            smsig_stxsbdt = tree_smsig.arrays(["stxs_bdtscore"], library="pd")
            acsig_stxsbdt = tree_acsig.arrays(["stxs_bdtscore"], library="pd")

            # Adding bdtstxs info into origin dataframe
            smsig_com = pd.concat([smsig, smsig_stxsbdt], axis=1)
            acsig_com = pd.concat([acsig, acsig_stxsbdt], axis=1)

            # Adding filter for different STXS bin
            smsig_com['sig/bkg'] = 0
            acsig_com['sig/bkg'] = 1

            # combine two dataframes into one.
            photondata_com = pd.concat([smsig_com, acsig_com], ignore_index=True, axis=0)
            print(photondata_com)

            # Seperate training dataset(70%) and testing dataset(15% for validation, and another 15% for testing)
            X, y = photondata_com.iloc[:, :-2], photondata_com['sig/bkg'] # drop bdtscore information
            X_train, X_tmp, y_train, y_tmp   = train_test_split(X, y, test_size=0.7, random_state=random.randint(0,42))
            X_valid, X_test, y_valid, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5)
            eval_set = [(X_train, y_train), (X_valid, y_valid)]


            # XGBoost sklearn configuration
            XGBEngine = xgb.XGBClassifier(
                objective        = 'binary:logistic',
                n_estimators     = 1000,
                max_depth        = 14,
                min_child_weight = 5,
                gamma            = 0, 
                #subsample        = 0.7, 
                #colsample_bytree = 0.9,
                learning_rate    = 0.05,
                reg_alpha        = 0,
                reg_lambda       = 0.5
            ) 

            # Training
            XGBEngine.fit(
                    X_train, y_train,
                    eval_metric           = ["auc"],
                    eval_set              = eval_set,
                    early_stopping_rounds = 10,
                    verbose               = True
            )

            # *Output model's xml for TMVA reader
            model = XGBEngine.get_booster().get_dump()
            Output_xml = 'output_{}_{}'.format(Prob_hist_title[channel_type][0], varset_name[setentry])
            convert_model(
                model,
                input_variables = featvarset_forxml[setentry],
                output_xml = '{}.xml'.format(Output_xml)
            )

            # Make prediction:
            y_pred = pd.DataFrame(XGBEngine.predict(X_test), columns=['sig/bkg'])
            y_pred_prob_train = pd.DataFrame(XGBEngine.predict_proba(X_train))
            y_pred_prob_valid = pd.DataFrame(XGBEngine.predict_proba(X_valid))
            y_pred_prob_test  = pd.DataFrame(XGBEngine.predict_proba(X_test))

            accuracy = metrics.accuracy_score(y_test, y_pred)
            predict_acc.append(accuracy)
            print('Accuracy: {:2.2%}'.format(accuracy))
            
            y_pred_com = pd.concat([y_pred_prob_test, y_test.reset_index(drop=True)], axis=1).dropna()
            print(y_pred_com)

            # *Make probabilities histograms
            mask_s = y_pred_com["sig/bkg"] == 1
            mask_b = y_pred_com["sig/bkg"] == 0

            df_hists = y_pred_com[mask_s]
            df_histb = y_pred_com[mask_b]
            bins = np.linspace(0., 1., 50)
            counts_err1, bins1 = np.histogram(1-df_histb[0], bins)

            # Drawing histogram
            plt.figure(figsize=(8,6))
            labels = [r"$f_{a1}=1$", r"$f_{a3}=1$"]
            bin_counts1, bin_edges1, patches1 = plt.hist(1-df_histb[0], bins, density=True, alpha=0.7, color='b', label=labels[0], log=True)
            bin_counts2, bin_edges2, patches2 = plt.hist(df_hists[1], bins, density=True, alpha=0.7, color='r', label=labels[1], log=True) 
            plt.ylim(0.05, 50)
            plt.xlabel("Probability", fontsize=14, loc='right') 
            plt.ylabel("Events", fontsize=14, loc='top') 
            plt.legend(bbox_to_anchor=(0.15, 1), prop={'size': 8})

            plotname = Prob_hist_title[channel_type][0]
            plt.savefig('BDTOuput_{}.pdf'.format(plotname))
            plt.clf()

            # *Plot ROC Curve
            # Train ROC
            fpr_train, tpr_train, threshold_train = metrics.roc_curve(y_train.values, y_pred_prob_train.values[:, 1], pos_label=1)
            roc_auc_train = metrics.auc(fpr_train, tpr_train)
            rfpr_train = np.asarray([ ( 1. - i ) for i in fpr_train ])
            # Test ROC
            fpr_test, tpr_test, threshold_test = metrics.roc_curve(y_test.values, y_pred_prob_test.values[:, 1], pos_label=1)
            roc_auc_test = metrics.auc(fpr_test, tpr_test)
            rfpr_test = np.asarray([(1. - i) for i in fpr_test])
            # Valid ROC
            fpr_valid, tpr_valid, threshold_valid = metrics.roc_curve(y_valid.values, y_pred_prob_valid.values[:, 1], pos_label=1)
            roc_auc_valid = metrics.auc(fpr_valid, tpr_valid)
            rfpr_valid = np.asarray([(1. - i) for i in fpr_valid])

            plt.figure(figsize=(8,6))
            plt.title(Prob_hist_title[channel_type][0])
            plt.plot(fpr_train, tpr_train,  'b', label = 'AUC (Train)  = %0.3f' % roc_auc_train)
            plt.plot(fpr_valid, tpr_valid, 'orange', label = 'AUC (Validate) = %0.3f' % roc_auc_valid)
            plt.plot(fpr_test, tpr_test, 'g', label = 'AUC (Test) = %0.3f' % roc_auc_test) 
            plt.legend(bbox_to_anchor=(1, 0.32))
            plt.plot([0, 1], [0, 1], '--')
            plt.ylabel('Signal efficiency', fontsize=10)
            plt.xlabel('Background efficiency', fontsize=10)
            plt.savefig('ROCCurve_{}.pdf'.format(Prob_hist_title[channel_type][0]), bbox_inches='tight')
            plt.clf()
            

if __name__ == "__main__":
    main()