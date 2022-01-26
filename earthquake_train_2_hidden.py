import modules.inference as inference
import modules.models_factory as models_factory
import time
import os
from collections import deque

num_of_layers = 5
num_of_types = 2

time_scale = 1e-3

time_log = time.time()
folder_name = (
    "/your/path/to/experiments/"  # replace this path with your path to store results
    + str(time_log)
    + "/"
)
os.mkdir(folder_name)

model_factory = models_factory.ModlesFactory(
    model_name="OEOT",
    data_path="/your/path/to/data_earthquake/",  # replace this path with your path to the data folder
    time_scale=time_scale,
)
model_factory.load_evidences("train", data_ratio=1, num_of_types=num_of_types)
model_factory.load_evidences("dev", data_ratio=1, num_of_types=num_of_types)

kernel_params_dq = deque(
    [
        0.6786821005630584,
        0.22879353021881102,
        0.01826475745966238,
        2.6613670203623108,
        0.7377663659680163,
        1.6773569754135238,
        15.260062664394468,
        0.3586926388156386,
        0.002648899490738625,
        1.6628332918788664,
        0.2861241794163845,
        0.002226073892788341,
    ]
)
virtual_kernel_params_dq = deque(
    [
        0.16381098299846472,
        0.2074614343203382,
        27.14575410333683,
        0.31021543478306324,
        0.6438698904968448,
        3.2479694005722077,
        0.07731751275895048,
        0.8377089962488387,
        38.197144500976464,
        0.723986915654927,
        0.33995758482280797,
        0.0736900266980569,
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
    track_validation=False,
    fix_kernel_params=False,
)
inference_EM.iterations_withoutparents()
