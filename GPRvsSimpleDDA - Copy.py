import numpy as np
import copy
import math
import random
import matplotlib.pyplot as plt
import RawData.DataBase as db
import KernalFuntion
from datetime import datetime
from scipy import stats

time_start_p = datetime.now()
def weight_calculation (input_x, output_y, kernal_matrix, observation_variance, mean_value):
    i_matrix_length = len(kernal_matrix)
    identity_matrix = np.eye(i_matrix_length)
    weight_matrix = np.matmul(np.linalg.inv(kernal_matrix + (observation_variance * identity_matrix)) , (output_y - mean_value))
    inv_matrix = np.linalg.inv(kernal_matrix + (observation_variance * identity_matrix))
    return weight_matrix, inv_matrix

# Hyperparameter Choice
loop_night = [7]
# Start Loop
for n_loop in loop_night:
    ComparsionList = []
    # Calculate the prior value
    prior_x = db.Level
    prior_y = db.PlayerLifeMean
    vs_Data = []
    # RBF
    # RBF Hypermeter Setting
    covariance = 1
    signal_variance = 1
    ls = n_loop
    # data simulation
    x = db.Level
    y = db.PlayerTime
    # Get all levels 101 levels
    y_all = []
    x_all = np.arange(0,101,1)
    x_all = list(x_all)
    # Multi Polynomial Function with Degree 2, all data simulation --> as the true data
    for i in y:
        Multi_Polynomial_Function = np.polyfit(x, i, 2)
        y_fit = np.polyval(Multi_Polynomial_Function, x_all)
        for ax in range (len(x)):
            y_fit[x[ax]] = i[ax]
        y_fit = np.round(y_fit)
        y_all.append(list(y_fit))
    # 4-fold cross validation
    for group124 in range (0,4):
        # log output
        print("This is Group -- " +str(group124 + 1)+", with RBF_ls = " +str(ls))
        print("    ")
        KernelName = "Kernel_p_"+str(group124)+"_RBF_"+str(ls)+".npy"
        LevelPoolName = "Level_p_"+str(group124)+"_RBF_"+str(ls)+".npy"
        PriorPredictionName = "Prediction_p_"+str(group124)+"_RBF_"+str(ls)+".npy"
        # Fold loop
        if group124 == 0:
            y_t1 = y_all[10:]
            y_t2 = y_all[:10]       
        elif group124 == 1:
            y_t1 = y_all[:10] + y_all[20:]
            y_t2 = y_all[10:20]
        elif group124 == 2:
            y_t1 = y_all[:20] + y_all[30:]
            y_t2 = y_all[20:30]
        else:
            y_t1 = y_all[:30]
            y_t2 = y_all[30:]
        # Determine the train and test data
        x_train = []
        y_train = []
        y_test = y_t2
        mk = np.array([0 for index in range (101)])
        # calculate the mean value
        for k in y_t1:
            mk = mk + np.array(k)
        mean_all = list(mk/len(y_t1))
        mean_v = mean_all
        # Train Data
        x_train = np.array([x_train]).T
        y_train = np.array([y_train]).T
        # Mean value
        mean_value = np.array([mean_v]).T
        # initial level pool
        print("    ")
        print("Do prior Prediction Lines")
        PT_x = []
        PT_y = []
        for i in range (0,30):
            PT_x = PT_x + x_all
            PT_y = PT_y + y_t1[i]
        # prior line
        Multi_Polynomial_Function = np.polyfit(PT_x, PT_y, 2)
        predict_p = np.polyval(Multi_Polynomial_Function, x_all)
        predict_p = np.round(predict_p)
        # Goal only by PT-Line
        IntermedianValue = (max(predict_p) + min(predict_p)) / 2
        Least_dis = min(abs(predict_p - IntermedianValue))
        level_pool = []
        for xxa in range (len(abs(predict_p - IntermedianValue))):
            if abs(predict_p - IntermedianValue)[xxa] == Least_dis:
                level_pool.append(copy.deepcopy(xxa))             

        LevelChoice_p = random.choice(level_pool)
        LevelNumber = x_all[LevelChoice_p]
        # Goal only by PT-Line
        IntermedianPriorValue = (max(predict_p) + min(predict_p))/2 
        level_pool.clear()   
        # calculate  time
        time_end_p = datetime.now()
        # logs
        print("Prior use " + str((time_end_p - time_start_p).seconds) +" seconds")
        # start DDA
        print("Body")
        # Body of DDA
        iter_r = []
        best_iter = []
        error_trend_all = []
        # calculate all players
        for i in range (len(y_test)):
            # recoder time
            time_start_indv = datetime.now()
            print("      ")
            print("Player Number " + str(i))
            # x train above
            x_train_n = x_train
            # y train above
            y_train_n = y_train
            # mean value above
            mean_value_n = mean_value
            # next level
            LevelChoice = LevelChoice_p
            # iteration time for each player
            ii = 0
            # player actual level choice
            playerDataSet_L = []
            # player actual performance
            playerDataSet_P = []
            # next level pool
            level_pool = []
            # validation and error
            m_error_trend = []
            # Goal for player only
            real_level_choice = []
            real_goal = (max(y_t2[i]) + min(y_t2[i])) / 2
            # Final goal
            real_goal = (IntermedianPriorValue + real_goal)/2
            # find the final recommendation level
            minvalue = min(abs(np.array(y_t2[i]) - real_goal))
            for i_h in range (len(abs(np.array(y_t2[i]) - real_goal))):
                if abs(np.array(y_t2[i]) - real_goal)[i_h] == minvalue:
                    real_level_choice.append(copy.deepcopy(i_h))
            # Create Player's own data
            P_Level_Seq = []
            P_Performance_Seq = []
            P_Mean_Seq = []
            # List that the level not be played
            CanbePlayedList = list(np.arange(0,101,1))
            mean_suppose_plus = predict_p
            # error stop
            e_stop = -1
            # player's iteration           
            while (True):
                print("      ")
                print("Start Iteration -- " + str(ii+1))
                ii = ii + 1               
                # player play the target level
                new_x = x_all[LevelChoice]
                # we get player's target performance
                new_y = y_t2[i][LevelChoice]
                print("Prior Max and min -- " + str(max(list(predict_p))) + " " + str(min(list(predict_p))))
                print("Player Max and min -- " + str(max(y_t2[i])) + " " + str(min(y_t2[i])))
                print("Real Goal -- " + str(real_goal))
                # record player's level
                playerDataSet_L.append(copy.deepcopy(LevelChoice))
                # record player's performance
                playerDataSet_P.append(copy.deepcopy(new_y))
                # new xtrain data
                P_Level_Seq = np.append(P_Level_Seq, np.array([new_x]))
                # new ytrain data
                P_Performance_Seq = np.append(P_Performance_Seq, np.array([new_y]))
                # correct the form
                P_Level_Seq = np.array([P_Level_Seq]).T
                P_Performance_Seq = np.array([P_Performance_Seq]).T
                # get corresponding mean value
                # P_Mean_Seq = np.append(P_Mean_Seq, [0])
                P_Mean_Seq = np.append(P_Mean_Seq, mean_value_n[LevelChoice])
                P_Mean_Seq = np.array([P_Mean_Seq]).T
                # Kernel Update RBF
                print("Kernel Update")
                #Weight, inv = weight_calculation (x_train_n, y_train_n, K_function, observation_variance = covariance, mean_value = mean_value_n)
                K_function_P = KernalFuntion.RBF_Self_K(P_Level_Seq, P_Level_Seq, signal_variance = signal_variance, length_scale = ls)
                Weight, inv = weight_calculation (P_Level_Seq, P_Performance_Seq, K_function_P, observation_variance = covariance, mean_value = P_Mean_Seq)
                # recod_p
                try:
                    recod_p = predict
                except:
                    recod_p = predict_p
                predict =[]
                for j in x_all:
                    # Input level
                    x_t = [[j]]
                    # Kernal Function
                    New_K = KernalFuntion.RBF_Self_K(x_t, P_Level_Seq, signal_variance = signal_variance, length_scale = ls)
                    # Mean Value
                    New_M = mean_v[j]
                    # Prediction
                    predicted_value = sum(New_K.T * Weight) + New_M
                    predicted_value = predicted_value[0]
                    predict.append(copy.deepcopy(predicted_value))
                # Get predicted value
                predict = list(np.round(predict))
                # Final Goal
                IntermedianValue = ((max(predict_p) + min(predict_p))/2 + (max(predict) + min(predict))/2) / 2
                print("Predict Goal -- " + str(IntermedianValue))
                # Calculated the abs distance
                Least_dis = min(abs(np.array(predict) - IntermedianValue))
                # Find the level good for the abs distance
                for xxa in range (len(abs(np.array(predict) - IntermedianValue))):
                    if abs(np.array(predict) - IntermedianValue)[xxa] == Least_dis:
                        level_pool.append(copy.deepcopy(xxa))
                # Lets choose a level.
                LevelChoice = random.choice(level_pool)
                flexable_range = list(np.arange(LevelChoice - 5,LevelChoice + 5, 1))
                print("Original Level Pool -- " + str(level_pool))
                while(True):
                    if LevelChoice in CanbePlayedList:
                        LevelChoice = LevelChoice
                        index_played = CanbePlayedList.index(LevelChoice)
                        CanbePlayedList.pop(index_played)
                        break
                    else:
                        index_played = level_pool.index(LevelChoice)
                        level_pool.pop(index_played)
                        if len(level_pool) == 0:
                            picked = False
                            for i_pick in flexable_range:
                                if i_pick in CanbePlayedList:
                                    print("Info: We will pick a near number")
                                    LevelChoice = i_pick
                                    level_pool.append(LevelChoice)
                                    picked = True
                                    break
                            if picked == False:
                                print("Warning: We will randomly pick a number here")
                                LevelChoice = random.choice(CanbePlayedList)
                                index_played = CanbePlayedList.index(LevelChoice)
                                CanbePlayedList.pop(index_played)
                                level_pool.append(LevelChoice)
                                break
                        else:
                            LevelChoice = random.choice(level_pool)
                print("Our Goal is -- " + str(real_level_choice) + str(real_goal))
                print("Level Pool -- " + str(level_pool) + str(level_pool[0]))
                print("Next Level is -- " + str(LevelChoice))
                performance_m_p = 0
                for x_pool in level_pool:
                    performance_m_p = performance_m_p + y_t2[i][x_pool]
                #performance_m_p = performance_m_p / len(level_pool)
                performance_m_p = y_t2[i][LevelChoice]
                real_p_m = y_t2[i][real_level_choice[0]]
                m_error = abs(performance_m_p - real_p_m)
                if e_stop == -1:
                    # Threshold
                    if m_error <= 0:
                        e_stop = m_error
                else:
                    m_error = e_stop
                m_error_trend.append(copy.deepcopy(m_error))
                level_pool.clear()
                print("iteration time -- "+str(ii))
                print("error value -- "+str(m_error))
                if ii >= 51:
                    error_trend_all.append(copy.deepcopy(m_error_trend))
                    print("Finish " + str(ii) +"'s player calculation have done")
                    break
            time_end_indv = datetime.now()
            print("Player " + str(i) + " use " + str((time_end_indv - time_start_indv).seconds) + " seconds for 51 iteration")
        for mx in error_trend_all:
            ComparsionList.append(copy.deepcopy(mx[-1]))

        error_trend_save = np.array(error_trend_all)
        file_name = "Trend_"+str(group124)+"_RBF_" + str(ls)
        np.save(file_name,error_trend_save)
        x_linshi = list(np.arange(0,51,1))
        slope_all = []
        for i in error_trend_all:
            slope,intercept,rvalue,pvalue,stderr = stats.linregress(x_linshi , i)
            slope_all.append(copy.deepcopy(slope))
        print("Model finess in Group " + str(group124 + 1) +" is:")
        vs_Data.append(copy.deepcopy(sum(slope_all)/len(slope_all)))
        print(str(vs_Data))
        slope_all_s = np.array(slope_all)
        file_name = "Slope_"+str(group124)+"_RBF_"+ str(ls)
        np.save(file_name,slope_all_s)
        print("--------------------------------------------------------------------------------")
        print("     ")