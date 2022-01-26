import modules.inference as inference
import modules.models_factory as models_factory
import time
import os
from collections import deque

num_of_layers = 7
num_of_types = 3

time_scale = 1e-3

time_log = time.time()
folder_name = (
    "/your/path/to/experiments/"  # replace this path with your path to store results
    + str(time_log) + "/"
)
os.mkdir(folder_name)

model_factory = models_factory.ModlesFactory(
    model_name="OEOT",
    data_path="/your/path/to/data_retweet/",  # replace this path with your path to the data folder
    time_scale=time_scale,
)
model_factory.load_evidences("train", data_ratio=1, num_of_types=3)
model_factory.load_evidences("dev", data_ratio=0.1, num_of_types=3)

kernel_params_dq = deque(
    [
        6.392672683889675,
        0.2101925747700868,
        0.2933375276445895,
        7.840819762047694,
        0.23338950613836595,
        1.0803227271224385,
        0.9324954670813387,
        0.09439042568122039,
        0.0020721781768240277,
        3.2968870817454734,
        0.12422475098409862,
        0.006502088354240421,
        2.1285404379405626,
        0.0965824851498811,
        0.0038105231357616312,
        1.963721719031124,
        0.11809520485986677,
        0.025243491717723686,
    ]
)

virtual_kernel_params_dq = deque(
    [
        0.060155026407675755,
        0.06852850037421147,
        61.93863623421051,
        0.06044999669806815,
        0.0819634218763341,
        22.16800211933966,
        0.16311406616092117,
        0.06614360539758511,
        107.87702978932927,
        0.3501044910145441,
        0.09853728003885541,
        0.37956956321533847,
        0.5931660514112671,
        0.10084794371742617,
        0.1606689110041322,
        0.9885869118513878,
        0.09745554349206664,
        0.0005866034010673854,
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
test_dpp_layers_events_factory = model_factory.build_model_factory(
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
    valid_sample_size=64,
    valid_sample_intervals=10,
    log_folder_name=folder_name,
    initial_burn_in_steps=4000,
    burn_in_steps=1000,
    valid_burn_in_steps=1,
    evidences=model_factory.evidences,
    valid_evidences=model_factory.valid_evidences,
    end_time_tuple=model_factory.end_time_tuple,
    valid_end_time_tuple=model_factory.valid_end_time_tuple,
    batch_size=1000,
    valid_batch_size=-1,
    dpp_layers_events_factory=dpp_layers_events_factory,
    valid_dpp_layers_events_factory=test_dpp_layers_events_factory,
    keep_last_samples=True,
    opt=True,
    num_of_events_evidence_each_example=model_factory.num_of_events_times_evidence,
    valid_num_of_events_evidence_each_example=model_factory.valid_num_of_events_times_evidence,
    time_scale=time_scale,
    alpha=0.01,
    virtual_alpha=0.01,
    optimization_method="adam",
    track_period=1,
    track_validation=True,
    fix_kernel_params=False,
)
inference_EM.iterations_withoutparents()
