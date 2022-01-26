import numpy as np
import pickle
import DPPLayers
from copy import deepcopy

def dpp_layers_factory(
    model_name,
    evidences,
    valid_evidences,
    num_of_types,
    num_of_layers,
    kernel_type,
    kernel_params_dq,
    virtual_kernel_params_dq,
    var_ids,
    virtual_var_ids,
    base_rate_init=0.08,
    valid_base_rate_init=0.08,
    virtual_base_rate_init=0.08,
):
    num_of_kernel_vars = len(var_ids)
    if model_name == "OT":
        prior_inference_DPPLayers = DPPLayers.DPPLayers()
        for i in range(num_of_layers):
            if i == 0:
                base_rate = np.array(
                    [base_rate_init for _ in range(len(evidences))]
                )
                valid_base_rate = np.array(
                    [valid_base_rate_init for _ in range(len(valid_evidences))]
                )
                top_layer = DPPLayers.TopLayer(
                    base_rate=deepcopy(base_rate),
                    virtual_base_rate=np.ones(len(evidences)) * virtual_base_rate_init,
                    valid_base_rate=deepcopy(valid_base_rate),
                    virtual_valid_base_rate=np.ones(len(valid_base_rate)) * virtual_base_rate_init,
                    kernels_type_lst=[kernel_type] * num_of_types,
                    kernels_param_lst=[
                        [
                            kernel_params_dq.popleft()
                            for _ in range(num_of_kernel_vars)
                        ]
                        for _ in range(num_of_types)
                    ],
                    var_ids=var_ids,
                    virtual_var_ids=virtual_var_ids,
                    layer_id=0,
                )
                prior_inference_DPPLayers.add_layer(
                    layer_id=0,
                    layer=top_layer,
                    parents_ids_dict={},
                    children_ids_dict={(i + 1): i for i in range(num_of_types)},
                )
            else:
                nontop_layer = DPPLayers.NonTopLayer(
                    base_rate=0.0,
                    virtual_base_rate=0.0,
                    kernels_type_lst=[],
                    virtual_kernels_type_lst=[kernel_type],
                    kernels_param_lst=[],
                    virtual_kernels_param_lst=[
                        [
                            virtual_kernel_params_dq.popleft()
                            for _ in range(num_of_kernel_vars)
                        ]
                    ],
                    layer_id=i,
                    var_ids=var_ids,
                    virtual_var_ids=virtual_var_ids,
                    synthetic_event_end=False,
                )
                prior_inference_DPPLayers.add_layer(
                    layer_id=i,
                    layer=nontop_layer,
                    parents_ids_dict={0: 0},
                    children_ids_dict={},
                )
    if model_name == "OEOT":
        prior_inference_DPPLayers = DPPLayers.DPPLayers()
        for i in range(num_of_layers):
            if i == 0:
                base_rate = np.array([base_rate_init for _ in range(len(evidences))])
                valid_base_rate = np.array(
                    [valid_base_rate_init for _ in range(len(valid_evidences))]
                )
                top_layer = DPPLayers.TopLayer(
                    base_rate=deepcopy(base_rate),
                    virtual_base_rate=deepcopy(
                        base_rate
                    ),  # np.zeros(len(model_factory.evidences)),
                    valid_base_rate=deepcopy(valid_base_rate),
                    virtual_valid_base_rate=deepcopy(valid_base_rate),
                    kernels_type_lst=[kernel_type] * num_of_types,
                    kernels_param_lst=[
                        [
                            kernel_params_dq.popleft()
                            for _ in range(num_of_kernel_vars)
                        ]
                        for _ in range(num_of_types)
                    ],
                    var_ids=var_ids,
                    virtual_var_ids=virtual_var_ids,
                    layer_id=0,
                )
                prior_inference_DPPLayers.add_layer(
                    layer_id=i,
                    layer=top_layer,
                    parents_ids_dict={},
                    children_ids_dict={(i+1): i for i in range(num_of_types)},
                )
            elif i < num_of_types + 1:
                nontop_layer = DPPLayers.NonTopLayer(
                    base_rate=0.0,
                    virtual_base_rate=0.0,
                    kernels_type_lst=[kernel_type],
                    # kernels_param_lst=[
                    #     [kernel_params[2 * (i - 1) + 6], kernel_params[2 * (i - 1) + 7]]
                    # ],
                    kernels_param_lst=[
                        [
                            kernel_params_dq.popleft()
                            for _ in range(num_of_kernel_vars)
                        ]
                    ],
                    virtual_kernels_type_lst=[kernel_type],
                    virtual_kernels_param_lst=[
                        [
                            virtual_kernel_params_dq.popleft()
                            for _ in range(num_of_kernel_vars)
                        ]
                    ],
                    var_ids=var_ids,
                    virtual_var_ids=virtual_var_ids,
                    layer_id=i,
                    synthetic_event_end=True,
                )
                prior_inference_DPPLayers.add_layer(
                    layer_id=i,
                    layer=nontop_layer,
                    parents_ids_dict={0: 0},
                    children_ids_dict={i + num_of_types: 0},
                )
            else:
                nontop_layer = DPPLayers.NonTopLayer(
                    base_rate=0.0,
                    virtual_base_rate=0.0,
                    kernels_type_lst=[],
                    kernels_param_lst=[],
                    virtual_kernels_type_lst=[kernel_type],
                    # virtual_kernels_param_lst=[
                    #     [1 / kernel_params[2 * (i - 4) + 6], kernel_params[2 * (i - 4) + 7]]
                    # ],
                    virtual_kernels_param_lst=[
                        [
                            virtual_kernel_params_dq.popleft()
                            for _ in range(num_of_kernel_vars)
                        ]
                    ],
                    var_ids=var_ids,
                    virtual_var_ids=virtual_var_ids,
                    layer_id=i,
                    # previous experiments use False
                    # synthetic_event_end=False,
                    synthetic_event_end=True,
                )
                prior_inference_DPPLayers.add_layer(
                    layer_id=i,
                    layer=nontop_layer,
                    parents_ids_dict={i - num_of_types: 0},
                    children_ids_dict={},
                )


    if model_name == "NOEOT":
        prior_inference_DPPLayers = DPPLayers.DPPLayers()
        for i in range(num_of_layers):
            if i == 0:
                base_rate = np.array([base_rate_init for _ in range(len(evidences))])
                valid_base_rate = np.array(
                    [valid_base_rate_init for _ in range(len(valid_evidences))]
                )
                top_layer = DPPLayers.TopLayer(
                    base_rate=deepcopy(base_rate),
                    virtual_base_rate=deepcopy(
                        base_rate
                    ),  # np.zeros(len(model_factory.evidences)),
                    valid_base_rate=deepcopy(valid_base_rate),
                    virtual_valid_base_rate=deepcopy(valid_base_rate),
                    kernels_type_lst=[kernel_type] * num_of_types,
                    kernels_param_lst=[
                        [
                            kernel_params_dq.popleft()
                            for _ in range(num_of_kernel_vars)
                        ]
                        for _ in range(num_of_types)
                    ],
                    var_ids=var_ids,
                    virtual_var_ids=virtual_var_ids,
                    layer_id=0,
                )
                prior_inference_DPPLayers.add_layer(
                    layer_id=i,
                    layer=top_layer,
                    parents_ids_dict={},
                    children_ids_dict={(i+1): i for i in range(num_of_types)},
                )
            elif i < num_of_types + 1:
                nontop_layer = DPPLayers.NonTopLayer(
                    base_rate=0.0,
                    virtual_base_rate=virtual_base_rate_init,
                    kernels_type_lst=[kernel_type],
                    kernels_param_lst=[
                        [
                            kernel_params_dq.popleft()
                            for _ in range(num_of_kernel_vars)
                        ]
                    ],
                    virtual_kernels_type_lst=[kernel_type],
                    virtual_kernels_param_lst=[
                        [
                            virtual_kernel_params_dq.popleft()
                            for _ in range(num_of_kernel_vars)
                        ]
                    ],
                    var_ids=var_ids,
                    virtual_var_ids=virtual_var_ids,
                    layer_id=i,
                    synthetic_event_end=True,
                )
                prior_inference_DPPLayers.add_layer(
                    layer_id=i,
                    layer=nontop_layer,
                    parents_ids_dict={0: 0},
                    children_ids_dict={i + num_of_types: 0},
                )
            elif i < num_of_layers - 2 * num_of_types:
                nontop_layer = DPPLayers.NonTopLayer(
                    base_rate=0.0,
                    virtual_base_rate=virtual_base_rate_init,
                    kernels_type_lst=[kernel_type],
                    kernels_param_lst=[
                        [
                            kernel_params_dq.popleft()
                            for _ in range(num_of_kernel_vars)
                        ]
                    ],
                    virtual_kernels_type_lst=[kernel_type],
                    virtual_kernels_param_lst=[
                        [
                            virtual_kernel_params_dq.popleft()
                            for _ in range(num_of_kernel_vars)
                        ]
                    ],
                    var_ids=var_ids,
                    virtual_var_ids=virtual_var_ids,
                    layer_id=i,
                    synthetic_event_end=True,
                )
                prior_inference_DPPLayers.add_layer(
                    layer_id=i,
                    layer=nontop_layer,
                    parents_ids_dict={i-num_of_types: 0},
                    children_ids_dict={i + num_of_types: 0},
                )
            elif i < num_of_layers - num_of_types:
                nontop_layer = DPPLayers.NonTopLayer(
                    base_rate=0.0,
                    virtual_base_rate=0.0,
                    kernels_type_lst=[kernel_type],
                    kernels_param_lst=[
                        [
                            kernel_params_dq.popleft()
                            for _ in range(num_of_kernel_vars)
                        ]
                    ],
                    virtual_kernels_type_lst=[kernel_type],
                    virtual_kernels_param_lst=[
                        [
                            virtual_kernel_params_dq.popleft()
                            for _ in range(num_of_kernel_vars)
                        ]
                    ],
                    var_ids=var_ids,
                    virtual_var_ids=virtual_var_ids,
                    layer_id=i,
                    synthetic_event_end=True,
                )
                prior_inference_DPPLayers.add_layer(
                    layer_id=i,
                    layer=nontop_layer,
                    parents_ids_dict={i-num_of_types: 0},
                    children_ids_dict={i + num_of_types: 0},
                )
            else:
                nontop_layer = DPPLayers.NonTopLayer(
                    base_rate=0.0,
                    virtual_base_rate=0.0,
                    kernels_type_lst=[],
                    kernels_param_lst=[],
                    virtual_kernels_type_lst=[kernel_type],
                    virtual_kernels_param_lst=[
                        [
                            virtual_kernel_params_dq.popleft()
                            for _ in range(num_of_kernel_vars)
                        ]
                    ],
                    var_ids=var_ids,
                    virtual_var_ids=virtual_var_ids,
                    layer_id=i,
                    synthetic_event_end=True,
                )
                prior_inference_DPPLayers.add_layer(
                    layer_id=i,
                    layer=nontop_layer,
                    parents_ids_dict={i - num_of_types: 0},
                    children_ids_dict={},
                )
    return prior_inference_DPPLayers


class ModlesFactory:
    def __init__(
        self,
        model_name,
        data_path,
        time_scale,
        num_of_oe=None,
        num_of_ot=None,
        num_of_hot=None,
        num_of_h_oeot=None,
        num_of_hoe=None,
    ):
        self.data_path = data_path  # train, dev, test
        self.model_name = model_name
        self.evidences = None
        self.valid_evidences = None
        self.test_evidences = None
        self.time_scale = time_scale
        self.num_of_oe = num_of_oe
        self.num_of_ot = num_of_ot
        self.num_of_hot = num_of_hot
        self.num_of_h_oeot = num_of_h_oeot
        self.num_of_hoe = num_of_hoe

    def load_data(self, data_type):
        with open(self.data_path + data_type + ".pkl", "rb") as f:
            data_temp = pickle.load(f, encoding="latin1")
            data = data_temp[data_type]
        return data

    def load_evidences(self, data_type, data_ratio, num_of_types, data_index_shift=0):
        data = self.load_data(data_type)
        num_of_evidences = int(len(data) * data_ratio)
        evidences = {}
        end_time_tuple = []
        num_of_events_times_evidence = []
        num_for_evidence_type = [0 for _ in range(num_of_types)]
        for i in range(len(data)):
            if i < num_of_evidences:
                evidences[i] = {}
                time_lst = [[] for _ in range(num_of_types)]
                i += data_index_shift
                for j in range(len(data[i])):
                    event_time = data[i][j]["time_since_start"]
                    event_type = data[i][j]["type_event"]
                    time_lst[event_type].append(event_time)
                    num_for_evidence_type[event_type] += 1
                i -= data_index_shift
                for k in range(len(time_lst)):
                    if self.model_name == "OT":
                        evidences[i][k + 1] = (
                            np.array(time_lst[k], dtype=np.float64) * self.time_scale
                        )
                    if self.model_name == "OEOT":
                        evidences[i][k + num_of_types + 1] = (
                            np.array(time_lst[k], dtype=np.float64) * self.time_scale
                        )
                    if self.model_name == "NOEOT":
                        evidences[i][k + self.num_of_oe * num_of_types + 1] = (
                            np.array(time_lst[k], dtype=np.float64) * self.time_scale
                        )
                end_time_tuple.append(
                    np.amax(np.concatenate(time_lst)) * self.time_scale
                )
                num_of_events_times_evidence.append(sum([len(e) for e in time_lst]))
        end_time_tuple = tuple(end_time_tuple)
        if data_type == "train":
            self.evidences = evidences
            self.end_time_tuple = end_time_tuple
            self.num_of_events_times_evidence = num_of_events_times_evidence
            self.num_for_evidence_type = num_for_evidence_type
        elif data_type == "dev":
            self.valid_evidences = evidences
            self.valid_end_time_tuple = end_time_tuple
            self.valid_num_of_events_times_evidence = num_of_events_times_evidence
            self.valid_num_for_evidence_type = num_for_evidence_type
        elif data_type == "test":
            self.test_evidences = evidences
            self.test_end_time_tuple = end_time_tuple
            self.test_num_of_events_times_evidence = num_of_events_times_evidence
            self.test_num_for_evidence_type = num_for_evidence_type

    def build_model_factory(self, data_type, dpp_layers):
        if data_type == "train":
            evidences = self.evidences
            end_time_tuple = self.end_time_tuple
            valid = False
        elif data_type == "dev":
            evidences = self.valid_evidences
            end_time_tuple = self.valid_end_time_tuple
            valid = True
        elif data_type == "test":
            evidences = self.test_evidences
            end_time_tuple = self.test_end_time_tuple
            valid = True

        def dpp_layers_events_factory(ex_id):
            dpp_layers_events = DPPLayers.DPPLayersEvents(
                dpp_layers=dpp_layers,
                ex_id=ex_id,
                valid=valid,
                end_time=end_time_tuple[ex_id],
                evidences_this_ex=evidences[ex_id],
            )
            return dpp_layers_events

        return dpp_layers_events_factory
