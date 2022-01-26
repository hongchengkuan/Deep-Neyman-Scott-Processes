import modules.models_factory as models_factory
from modules.predict_time_type import predict_time_type
import time
import os
import argparse
from collections import deque

num_of_layers = 3
num_of_types = 2
model_name = "OT"
time_scale = 1e-3
# num of evidences 53

model_factory = models_factory.ModlesFactory(
    model_name=model_name,
    data_path="/your/path/to/data_earthquake/",  # replace this path with your path to the data folder
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
    kernel_params_dq=deque(
        [
            4.246983398733107,
            0.42863428555288735,
            0.012524744708947239,
            2.639001147460542,
            0.33765259760578265,
            0.008554339304444733,
        ]
    ),
    virtual_kernel_params_dq=deque(
        [
            0.14029351376141674,
            0.34588603547638186,
            0.22236071670161112,
            0.17943741827915724,
            0.28514723577727574,
            0.4196774359691575,
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
