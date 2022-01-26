import modules.models_factory as models_factory
from modules.predict_time_type import predict_time_type
import time
import os
import argparse
from collections import deque

num_of_layers = 4
num_of_types = 3
model_name = "OT"
time_scale = 1e-3

model_factory = models_factory.ModlesFactory(
    model_name=model_name,
    data_path="/your/path/to/data_retweet/",  # replace this path with your path to the data folder
    time_scale=time_scale,
)
time_log = time.time()
parser = argparse.ArgumentParser(description="predict model ... ")

parser.add_argument("-e", "--ExampleId", required=True)
args = parser.parse_args()
log_folder_name = (
    "/your/path/to/experiments/"  # replace this path with your path to store results 
    + args.ExampleId + "/"
)
os.mkdir(log_folder_name)
model_factory.load_evidences("test", data_ratio=1.0, num_of_types=3)
predict_time_type(
    model_name=model_name,
    evidence_dict=model_factory.test_evidences[int(args.ExampleId)],
    num_of_types=num_of_types,
    num_of_layers=num_of_layers,
    kernel_type="GammaKernel",
    base_rate_init=0.5,
    valid_base_rate_init=0.08,
    virtual_base_rate_init=0,
    kernel_params_dq=deque(
        [
            6.006670540797534,
            0.15210199948301395,
            0.013832555299523059,
            5.5097042944943055,
            0.11551755958350977,
            0.01197647392170633,
            0.5578397633996484,
            0.1258478208755482,
            0.010855449955177112,
        ]
    ),
    virtual_kernel_params_dq=deque(
        [
            0.07032797478075346,
            0.050237579678172044,
            0.7734485112129983,
            0.11043083655864504,
            0.04795724310949159,
            8.622112898537514,
            0.16673568199179242,
            0.07462781002002578,
            12.943470161228968,
        ]
    ),
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

