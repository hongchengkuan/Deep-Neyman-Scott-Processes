import modules.inference as inference
import modules.models_factory as models_factory
import time
import os
from collections import deque

num_of_layers = 6
num_of_types = 5

time_scale = 1e-3

time_log = time.time()
folder_name = (
    "/your/path/to/experiments/"  # replace this path with your path to store results 
    + str(time_log) + "/"
)
os.mkdir(folder_name)

model_factory = models_factory.ModlesFactory(
    model_name="OT",
    data_path="/your/path/to/data_homicide/",  # replace this path with your path to the data folder
    time_scale=time_scale,
)
model_factory.load_evidences("train", data_ratio=1.0, num_of_types=num_of_types)
model_factory.load_evidences("dev", data_ratio=1.0, num_of_types=num_of_types)
kernel_params_dq = deque(
    [
        0.2549860427667309,
        0.1130057611929039,
        4.448531535098438e-07,
        0.16773255693560382,
        0.11039960569439906,
        4.7653838067625856e-07,
        0.16500682084270257,
        0.08987553377313669,
        6.17677859944218e-07,
        0.21363562179503198,
        0.1092408006711863,
        3.6174298341994944e-07,
        0.17019416334668092,
        0.1346106629381687,
        8.419040756486973e-05,
    ]
)
virtual_kernel_params_dq = deque(
    [
        2.5515972725958926,
        0.14179279689463867,
        2.8198204214972227e-07,
        2.5564952550935605,
        0.15840681039292215,
        5.059467133204954e-07,
        2.211378196093862,
        0.12331005083561654,
        3.8486532904178205e-07,
        3.0330685114674965,
        0.16764888638765957,
        4.0687324736584853e-07,
        3.4938960539775037,
        0.18106623705869013,
        6.616076194302412e-07,
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
