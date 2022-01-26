import modules.models_factory as models_factory
from modules.predict_time_type import predict_time_type
import time
import os
import argparse
from collections import deque

num_of_layers = 6
num_of_types = 5
model_name = "OT"
time_scale = 1e-3

model_factory = models_factory.ModlesFactory(
    model_name=model_name,
    data_path="/your/path/to/data_homicide/",  # replace this path with your path to the data folder
    time_scale=time_scale,
)
time_log = time.time()
parser = argparse.ArgumentParser(description="predict model ... ")

parser.add_argument("-e", "--ExampleId", required=True)
args = parser.parse_args()
log_folder_name = (
    "/your/path/to/experiments/"  # replace this path with your path to store results
    + args.ExampleId 
    + "/"
)
os.mkdir(log_folder_name)
model_factory.load_evidences("test", data_ratio=1.0, num_of_types=num_of_types)
predict_time_type(
    model_name=model_name,
    evidence_dict=model_factory.test_evidences[int(args.ExampleId)],
    num_of_types=num_of_types,
    num_of_layers=num_of_layers,
    kernel_type="GammaKernel",
    base_rate_init=0.5,
    valid_base_rate_init=0.08,
    virtual_base_rate_init=0,
    kernel_params_dq = deque([
        0.25457340200601775,
        0.1156590645970127,
        4.3754281618002837e-07,
        0.1743401125921959,
        0.10823613583002617,
        3.2019832232993947e-07,
        0.16982158924243132,
        0.09756520542843779,
        4.1745808556114426e-07,
        0.2115661756239163,
        0.10784254585867298,
        3.0016967269756784e-07,
        0.16887390113079523,
        0.15420227977589815,
        8.907083240925036e-05,
    ]),
    virtual_kernel_params_dq = deque([
        3.6122695338388766,
        0.14983691506243796,
        1.3833164056683804e-07,
        2.775169057076011,
        0.13186550994921176,
        1.0233935752526621e-07,
        2.062435914655161,
        0.12984553225877007,
        1.9607708000064514e-07,
        3.5491386445231785,
        0.15289194127616,
        7.416167293943342e-08,
        4.696657810158091,
        0.20057232100516206,
        3.4246160105699907e-07,
    ]),
    var_ids=[1, 2, 3],
    virtual_var_ids=[0, 1, 2, 3],
    initial_burn_in_steps=0,
    burn_in_steps=0,
    num_of_mcmc_samples=250,
    keep_last_samples=True,
    time_scale=time_scale,
    sample_intervals=10,
    random_sample_layer=True,
    log_folder_name=log_folder_name,
    em_iters=5,
)
