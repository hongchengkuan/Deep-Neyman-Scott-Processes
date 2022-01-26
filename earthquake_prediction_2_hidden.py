import modules.models_factory as models_factory
from modules.predict_time_type import predict_time_type
import time
import os
import argparse
from collections import deque

num_of_layers = 5
num_of_types = 2
model_name = "OEOT"
time_scale = 1e-3

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
            1.0283417271088702,
            0.539638917059432,
            3.9610157287514376,
            1.613411401374775,
            0.5463088924537995,
            7.8794177049966665,
            5.39658193645925,
            0.38513977695741447,
            0.003230320943242581,
            2.0594736470286623,
            0.3085757040597056,
            0.003016582671566066,
        ]
    ),
    virtual_kernel_params_dq=deque(
        [
            0.2882138117938841,
            0.39822977502993556,
            6.985465955432222,
            0.38046805102149234,
            0.4404439946393564,
            14.59210941671301,
            0.22353116537385304,
            0.3645183009057448,
            0.18455328744115562,
            0.5738869066542155,
            0.3021890193902747,
            0.06872419405155102,
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
