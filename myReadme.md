ctr_funcs:
    cal_auc(pred_score, label):
    cal_rmse(pred_score, label):
    cal_rectified_rmse(pred_score, label, sample_rate):
    list_flatten(input_list):
    count_lines(file_name):
    tf_read_data(file_name_queue, label_col_idx, record_defaults):
    tf_read_data_wo_label(file_name_queue, record_defaults):
    tf_input_pipeline(file_names, batch_size, num_epochs=1, label_col_idx=0,        
                             record_defaults=record_defaults):
    tf_input_pipeline_wo_label(file_names, batch_size, num_epochs=1, 
                             record_defaults=record_defaults):
    tf_input_pipeline_test(file_names, batch_size, num_epochs=1, label_col_idx=0, 
                             record_defaults=record_defaults):
    tfrecord_input_pipeline(file_names, num_csv_col, batch_size, num_epochs=1):
    tfrecord_input_pipeline_test(file_names, num_csv_col, batch_size, num_epochs=1):
    print_time():

minet_para_tune:
'''
Code logic: read config -> load data -> define functions -> define placeholders and variables 
-> define computation graph -> launch computation graph (run session)
-> train -> test -> print results
'''
    