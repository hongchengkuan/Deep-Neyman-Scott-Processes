import math

type_pred_correct_num = 0
error_sum = 0
ll_sum = 0
pred_tot_num = 0
time_scale = 1e-3

num_of_sequences = 2000  # 2000 for retweet, 2 for homicide, 53 for earthquake
for i in range(0, num_of_sequences):
    file_name = (
        "/your/path/to/experiments/of/prediction/"  # replace this path with your path to the results of prediction
        + str(i) + "/prediction.log"
    )
    f = open(file_name, "r")
    for count, line in enumerate(f):
        if count == 0:
            type_pred_correct_num += int(line)
        if count == 1:
            error_sum += float(line)
        if count == 2:
            ll_sum += float(line)
        if count == 5:
            pred_tot_num += int(line)
print("type accuracy = ", type_pred_correct_num / pred_tot_num)
print("rmse = ", math.sqrt(error_sum / pred_tot_num) / time_scale)
print("ll = ", ll_sum / (pred_tot_num + num_of_sequences) + math.log(time_scale))
