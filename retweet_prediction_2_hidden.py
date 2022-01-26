import modules.models_factory as models_factory
from modules.predict_time_type import predict_time_type
import time
import os
import argparse
from collections import deque

num_of_layers = 7
num_of_types = 3
model_name = "OEOT"
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
            4.696462950493177,
            0.18944420274909307,
            0.2782119067361892,
            8.614039058213207,
            0.24567300668640396,
            1.0393573232868647,
            1.0237777766367133,
            0.09161522693674286,
            0.003668407014459594,
            3.6315799179272084,
            0.13043326874548172,
            0.007119017387232255,
            1.8272833063409633,
            0.0944382061484448,
            0.004616543837440334,
            1.5813645552609599,
            0.12058590133308593,
            0.022368736080158335,
        ]
    ),
    virtual_kernel_params_dq=deque(
        [
            0.07202998239514934,
            0.06884400140866016,
            140.08441378512867,
            0.052965094396001604,
            0.10172627834927774,
            80.97019116515426,
            0.18944594916895066,
            0.07195833157490096,
            187.98977588503968,
            0.31869329243261507,
            0.09596546606708226,
            0.24577213987243418,
            0.6749321051771368,
            0.10309544114143328,
            0.15926085430671263,
            1.523275822305071,
            0.09988882991355726,
            2.924651240525741e-06,
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
