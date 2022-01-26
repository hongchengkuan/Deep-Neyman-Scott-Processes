import modules.inference as inference
import modules.models_factory as models_factory
import time
import os
from collections import deque

num_of_layers = 3
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
    model_name="OT",
    data_path="/your/path/to/data_earthquake/",  # replace this path with your path to the data folder
    time_scale=time_scale,
)
model_factory.load_evidences("train", data_ratio=1.0, num_of_types=num_of_types)
model_factory.load_evidences("dev", data_ratio=1.0, num_of_types=num_of_types)
kernel_params_dq = deque(
    [
        8.346547914292966,
        0.39854391507310605,
        0.005784269713517436,
        4.010488951820977,
        0.32423860264537246,
        0.0026503779019187534,
    ]
)
virtual_kernel_params_dq = deque(
    [
        0.0847596473486434,
        0.31816241530884576,
        0.46869026647354023,
        0.11014326177459212,
        0.2744649754562491,
        0.9520985809077167,
    ]
)
kernel_type = "GammaKernel"
var_ids = [1, 2, 3]
virtual_var_ids = [0, 1, 2, 3]

prior_inference_DPPLayers = models_factory.dpp_layers_factory(
    model_name="OT",
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
    virtual_base_rate_init=0.,
)


dpp_layers_events_factory = model_factory.build_model_factory(
    data_type="train", dpp_layers=prior_inference_DPPLayers,
)
valid_dpp_layers_events_factory = model_factory.build_model_factory(
    data_type="dev", dpp_layers=prior_inference_DPPLayers,
)


print(f" num of events in evidence = {sum(model_factory.num_of_events_times_evidence)}")
inference_EM = inference.EM(
    dpp_layers=prior_inference_DPPLayers,
    num_of_epochs=4000,
    sample_size=64,
    sample_intervals=10,
    valid_sample_size=10,
    valid_sample_intervals=40,
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
