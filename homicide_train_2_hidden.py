import modules.inference as inference
import modules.models_factory as models_factory
import time
import os
from collections import deque

num_of_evidences = 20000
num_of_layers = 11
num_of_types = 5

time_scale = 1e-3

time_log = time.time()
folder_name = (
    "/your/path/to/experiments/"  # replace this path with your path to store results  
    + str(time_log) + "/"
)
os.mkdir(folder_name)

model_factory = models_factory.ModlesFactory(
    model_name="OEOT",
    data_path="/your/path/to/data_homicide/",  # replace this path with your path to the data folder
    time_scale=time_scale,
)
model_factory.load_evidences("train", data_ratio=1, num_of_types=num_of_types)
model_factory.load_evidences("dev", data_ratio=1, num_of_types=num_of_types)

kernel_params_dq = deque(
    [
        0.9989633333677028,
        0.345467708932763,
        4.9239606734291224e-08,
        0.5133564288790535,
        0.3048411860398642,
        5.4897166415768986e-08,
        0.43502866408180707,
        0.34303062323735656,
        4.871111373305559e-08,
        0.7028802285939566,
        0.32833470794181435,
        3.8616477935802956e-08,
        0.26329989113286184,
        0.19795401208173927,
        2.969566460481817e-06,
        0.41423560500398177,
        0.10876358102712658,
        0.00010783249730984486,
        0.40926714399656977,
        0.10514378577713622,
        6.803020240208032e-05,
        0.7180287296702794,
        0.10196311728458707,
        2.684424678632667e-05,
        0.6118759773929557,
        0.1028324392539396,
        2.1492164223055233e-06,
        0.27587367588030504,
        0.1432490082455765,
        0.00044432001000651905,
    ]
)
virtual_kernel_params_dq = deque(
    [
        0.4613450929386201,
        0.4009675420272256,
        0.0001268730993200981,
        0.3520524799981143,
        0.3059687196725334,
        0.0006716158448649349,
        0.4351374986174005,
        0.39980873967212593,
        0.0008394433956482962,
        6.699373746592573,
        6.5123664189303065,
        8.254865303227116e-05,
        0.5548704888035515,
        0.19129627810225142,
        0.0008863509750916961,
        2.890671910964869,
        0.1884898225179219,
        1.3393747296868933e-05,
        3.2799532282723627,
        0.19669790750782973,
        7.881997992855629e-06,
        1.9981965682099772,
        0.15051275859967847,
        3.415926983211305e-06,
        2.9697421599771556,
        0.17736247940916267,
        6.10888263422744e-06,
        5.49566516812685,
        0.24470402114009812,
        5.398759859288701e-06,
    ]
)
kernel_type = "GammaKernel"
var_ids = [1, 2, 3]
virtual_var_ids = [0, 1, 2, 3]

prior_inference_DPPLayers = models_factory.dpp_layers_factory(
    model_name="OEOT",
    evidences=model_factory.evidences,
    valid_evidences=model_factory.valid_evidences,
    num_of_types=num_of_types,
    num_of_layers=num_of_layers,
    kernel_type=kernel_type,
    kernel_params_dq=kernel_params_dq,
    virtual_kernel_params_dq=virtual_kernel_params_dq,
    var_ids=var_ids,
    virtual_var_ids=virtual_var_ids,
    base_rate_init=0.08,
    valid_base_rate_init=0.08,
    virtual_base_rate_init=0.08,
)

dpp_layers_events_factory = model_factory.build_model_factory(
    data_type="train", dpp_layers=prior_inference_DPPLayers,
)
valid_dpp_layers_events_factory = model_factory.build_model_factory(
    data_type="dev", dpp_layers=prior_inference_DPPLayers,
)


print(f" num of events in evidence = {sum(model_factory.num_of_events_times_evidence)}")
print(
    f" num of events in valid evidence = {sum(model_factory.valid_num_of_events_times_evidence)}"
)
inference_EM = inference.EM(
    dpp_layers=prior_inference_DPPLayers,
    num_of_epochs=4000,
    sample_size=64,
    sample_intervals=10,
    valid_sample_size=10,
    valid_sample_intervals=10,
    log_folder_name=folder_name,
    initial_burn_in_steps=0,
    burn_in_steps=0,
    valid_burn_in_steps=1,
    evidences=model_factory.evidences,
    valid_evidences=model_factory.valid_evidences,
    end_time_tuple=model_factory.end_time_tuple,
    valid_end_time_tuple=model_factory.valid_end_time_tuple,
    batch_size=-1,
    valid_batch_size=-1,
    dpp_layers_events_factory=dpp_layers_events_factory,
    valid_dpp_layers_events_factory=valid_dpp_layers_events_factory,
    keep_last_samples=True,
    opt=True,
    num_of_events_evidence_each_example=model_factory.num_of_events_times_evidence,
    valid_num_of_events_evidence_each_example=model_factory.valid_num_of_events_times_evidence,
    time_scale=time_scale,
    alpha=0.01,
    virtual_alpha=0.1,
    optimization_method="adam",
    track_period=2,
    track_validation=True,
    fix_kernel_params=False,
)
inference_EM.iterations_withoutparents()
