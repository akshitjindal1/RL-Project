# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'video_summ.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


# from PyQt5 import QtCore, QtGui, QtWidgets
import os
from gui_main_policy_long_vid_interactive_without_feedback import generate_normal_summary
from gui_main_policy_long_vid_interactive_with_feedback import generate_summary_with_feedback
from gui_generate_video_from_summary import generate_video_from_summary
from gui_histogram_clustering import plot_normal_and_custom_summary

def secs_to_indexes(temp_in, idxes):
    all_idxes = []
    for row in temp_in:
        print (60*int(row[0:2]) + int(row[3:5]), 60*int(row[6:8])+ int(row[9:11]))
        temp_idx = range(60*int(row[0:2]) + int(row[3:5]), 60*int(row[6:8])+ int(row[9:11]))
        all_idxes = all_idxes + list(temp_idx)
    return all_idxes

def write_feedback_indexes(file_name, positive_feedback_idxes):
    index_file = open(file_name,"w")
    for idx in positive_feedback_idxes:
        index_file.write(str(idx))
        index_file.write("\n")


dataset_global = []
video_name_global = []

def btn_click_generate_summary_without_feedback(dataset, video_name):
    """ Helper function to generate summary without feedback for the selected dataset and video
    call the python script : main_policy_long_vid_interactive_without_feedback.py with all the necessary flags
    """
    if (dataset == "" or video_name == ""):
        print("Please select dataset and video_name")
        return -1
    else:
        normal_summary_path = generate_normal_summary(dataset, video_name)
        #normal_summary_path = 'output_summary_without_feedback/Alin_Day_1.mp4_Disney_policy_grad_summary_length_600_subshot_size_200_hidden_dim_256_summary_without_feedback.txt'

        generate_video_from_summary(normal_summary_path, dataset, video_name) # you can comment this line if don't want to generate video otherwise give the video path in 'gui_generate_video_from_summary.py' file to generate summary video.
        print ('Normal Summary Generated:  ', normal_summary_path)
        return 0


def btn_click_generate_summary_with_feedback():
    """This is a function to trigger the button click event for generating summary with a given feedback
    The button click will read the values entered in the text box for positive and negative feedback
    save it, and will then use it later
    """
    positive_feed = textEdit_possitive_feedback_interval.toPlainText()
    negative_feed = textEdit_negative_feedback_interval.toPlainText()

    f_ptr = open(normal_summary_path, 'r')
    summary_one_hot = []
    for i in f_ptr:
        summary_one_hot.append(i[:-1])
    idxes = [idx for idx, val in enumerate(summary_one_hot) if val == '1']
    #print (idxes)
    positive_feed_list = positive_feed.split('\n')
    negative_feed_list = negative_feed.split('\n')
    positive_feedback_idxes = secs_to_indexes(positive_feed_list, idxes)
    negative_feedback_idxes = secs_to_indexes(negative_feed_list, idxes)
    positive_feedback_idxes_to_vid = list([val for idx, val in enumerate(idxes) if idx in positive_feedback_idxes])
    negative_feedback_idxes_to_vid = list([val for idx, val in enumerate(idxes) if idx in negative_feedback_idxes])

    customized_summary_path = generate_summary_with_feedback(dataset, video_name, normal_summary_path, positive_feedback_idxes_to_vid, negative_feedback_idxes_to_vid)
    #customized_summary_path = 'output_summary_with_feedback/Alin_Day_1.mp4_Disney_policy_grad_summary_length_600_subshot_size_200_hidden_dim_256_summary_with_feedback_krishan.txt'
    generate_video_from_summary(customized_summary_path, dataset, video_name) # you can comment this line if don't want to generate video otherwise give the video path in 'gui_generate_video_from_summary.py' file to generate summary video.
    plot_normal_and_custom_summary(normal_summary_path, customized_summary_path)
    print ('---------------Customized Summary Generated-------------------------')


        # normal_summary = '/media/enigma/f0762f3b-20d1-42a7-9fe1-60385c4a8a3e/video_summarization/actor_critic/code_interactive_summ/output_summary_without_feedback/'
        # write_feedback_path = '/media/enigma/f0762f3b-20d1-42a7-9fe1-60385c4a8a3e/video_summarization/actor_critic/code_interactive_summ/gui_feedback/'
        # files = os.listdir(normal_summary)
        # for my_file in files:
        #     if '.txt' in my_file:
        #         f_ptr = open(normal_summary + my_file, 'r')
        #         summary_one_hot = []
        #         for i in f_ptr:
        #             summary_one_hot.append(i[:-1])
        #         idxes = [idx for idx, val in enumerate(summary_one_hot) if val == '1']
        #         print (idxes)
        #         positive_feed_list = positive_feed.split('\n')
        #         negative_feed_list = negative_feed.split('\n')
        #         positive_feedback_idxes = secs_to_indexes(positive_feed_list, idxes)
        #         negative_feedback_idxes = secs_to_indexes(negative_feed_list, idxes)
        #         write_feedback_indexes(write_feedback_path + 'positive_feedback.txt', list([val for idx, val in enumerate(idxes) if idx in positive_feedback_idxes]))
        #         write_feedback_indexes(write_feedback_path + 'negative_feedback.txt', list([val for idx, val in enumerate(idxes) if idx in negative_feedback_idxes]))
        # call_python_function_feedback = 'python gui_main_policy_long_vid_interactive_with_feedback.py -d datasets/'+ dataset_global +'_features.h5 -s datasets/'+ dataset_global +'_splits.json -m summe --gpu 0 --save-dir log/' + video_name_global + '_int_pol-split0 --split-id 0 --verbose'
        # #os.system(call_python_function_feedback)
        # temp_py_script = 'python generate_video_from_summary.py -d '+ dataset_global +' -v '+ video_name_global +' -s feedback'
        # os.system(temp_py_script)
        # print ('---------------Customized Summary Generated-------------------------')


def main():
    
    run = True

    while run:

        print ('================================')
        print("Available Datsets:")
        print(
            "1.UTE\n"
        )
        print("\nAvailable Video Files:")
        print(
            "1. P01\n2. P02\n3. P03\n4. P04\n"
        )

        # dataset_name = input("Please Enter the Dataset Name: ")
        # video_name = input("Please Enter the Video Name: ")
        
        dataset_name = "UTE"
        video_name = "P03"
        
        print('================================\n')
        success = btn_click_generate_summary_without_feedback(dataset_name, video_name)

        if success == 0:
            break


if __name__ == "__main__":
    main()
