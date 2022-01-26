import modules.inference as inference
import modules.models_factory as models_factory
from modules.predict_time_type import predict_time_type
import time
import os
import argparse
from collections import deque

num_of_layers = 11
num_of_types = 5
model_name = "OEOT"
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
    kernel_params_dq=deque(
        [
            0.32855715948307157,
            0.19843585757876292,
            1.541815796473124e-08,
            0.12138570061400421,
            0.18690251472065017,
            1.5067733169784393e-08,
            0.08525085640711669,
            0.17851172519643027,
            1.311110900135425e-08,
            0.2736454478836442,
            0.24840336026733362,
            1.4583628864768759e-08,
            0.10275887132219398,
            0.19187377648229043,
            3.4214694702728854e-06,
            0.26438001352890494,
            0.11328057769671107,
            0.00030566886501111264,
            0.501879972075598,
            0.11834124215317277,
            2.3710255834621617e-05,
            0.7997928734156814,
            0.09775129971105977,
            8.21796200367804e-06,
            0.4099289994356742,
            0.12759575076499616,
            5.123391578097139e-05,
            0.31822882957385656,
            0.15410179109407326,
            0.00036668489533059063,
        ]
    ),
    virtual_kernel_params_dq=deque(
        [
            0.5765997821695339,
            0.1978527049127138,
            0.0003441142154089868,
            0.6506462768427429,
            0.18241822408740951,
            0.00035628683241626963,
            0.7294518746115131,
            0.18895533692041908,
            0.0003583549050002589,
            6.744915131800051,
            6.990692085003744,
            7.813614084117054e-05,
            0.6764624680789119,
            0.19904926639248288,
            0.00034332231981897015,
            6.141943560137988,
            0.21794649641088162,
            2.373389954712241e-06,
            6.36590163298271,
            0.2087839935774346,
            1.21251562322604e-07,
            4.183430041646454,
            0.14410493440404618,
            1.573282203240616e-08,
            5.827256318052873,
            0.21697373396514452,
            5.527890933675083e-07,
            4.147849305156284,
            0.2531906913134637,
            1.4181348925594532e-05,
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

