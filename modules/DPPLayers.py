import numpy as np
from collections import deque, namedtuple
from scipy.special import gamma, gammainc, gammaincinv, gammaincc
import copy

copy_deepcopy = copy.deepcopy


np_sum, np_exp = np.sum, np.exp
np_log = np.log
np_array = np.array
np_where = np.where
np_logical_and = np.logical_and
np_inf = np.inf
np_log1p = np.log1p
np_min = np.min
np_abs = np.abs
np_maximum = np.maximum
np_max = np.max
np_append = np.append
np_concatenate = np.concatenate
np_mean = np.mean
np_sort = np.sort

Stats_collections = namedtuple(
    "stats",
    [
        "top_ll",
        "top_mu_grad",
        "top_base_rate",
        "ll",
        "grad",
        "virtual_top_ll",
        "virtual_top_ll_wrt_real",
        "virtual_top_grad_mu",
        "virtual_top_grad_mu_wrt_real",
        "virtual_ll",
        "virtual_ll_wrt_real",
        "virtual_grad",
        "virtual_grad_wrt_real",
        "reduced_var_top_mu_grad",
        "reduced_var_top_virtual_mu_grad",
        "reduced_var_grad",
        "reduced_var_virtual_grad",
    ],
)



class SoftPlus:
    def __init__(self):
        pass

    @staticmethod
    def fun(x):
        return np_log(1 + np_exp(-np_abs(x))) + np.maximum(x, 0)
        # return np_log(1 + np.exp(x))

    @staticmethod
    def grad_fun(x):
        if hasattr(x, "__len__"):
            x[x > 100] = 100
            result = np_exp(x) / (1 + np_exp(x))
        else:
            result = np_exp(x) / (1 + np_exp(x)) if x <= 100 else 1
        return result

    @staticmethod
    def inv_fun(x):
        # return np_log(np_exp(x) - 1)
        assert not np.any(x < 0)
        if hasattr(x, "__len__"):
            x[x == 0] = np_inf
        elif x == 0:
            return -np_inf
        return np_where(x == np_inf, -np_inf, np_log(1 - np_exp(-x)) + x)

    @staticmethod
    def grad_inv_fun(x):
        if hasattr(x, "__len__"):
            x[x > 100] = 100
            result = (np_exp(x) - 1) / np_exp(x)
        else:
            result = (np_exp(x) - 1) / np_exp(x) if x <= 100 else 1
        return result
        # return (np_exp(x) - 1) / np_exp(x)


sp = SoftPlus()


class DPPLayers:
    """
    This is a class for getting and assigning parameters for each layer.
    """
    def __init__(self):
        self.layers = []
        self.parents_ids_lst = []
        self.children_ids_lst = []
        self.evidences_ids_set = frozenset()
        self.nontop_ids_lst = []
        self.top_ids_lst = []

    def add_layer(self, layer_id, layer, parents_ids_dict, children_ids_dict):
        # ids_dict {layer_id:stats_lst_id}
        self.layers.append(layer)
        self.parents_ids_lst.append(parents_ids_dict)
        self.children_ids_lst.append(children_ids_dict)
        if not children_ids_dict:
            self.evidences_ids_set = self.evidences_ids_set.union([layer_id])
        if not parents_ids_dict:
            self.top_ids_lst.append(layer_id)
        else:
            self.nontop_ids_lst.append(layer_id)

    def get_top_mu(self, example_ids, valid, transform=True):
        layers = self.layers
        top_ids_lst = self.top_ids_lst
        if valid:
            if transform:
                base_rate_arr = [
                    sp.inv_fun(layers[layer_id].valid_base_rate[example_ids])
                    for layer_id in top_ids_lst
                ]
            else:
                base_rate_arr = [
                    layers[layer_id].valid_base_rate[example_ids]
                    for layer_id in top_ids_lst
                ]
            return np.concatenate(base_rate_arr)
        else:
            if transform:
                base_rate_arr = [
                    sp.inv_fun(layers[layer_id].base_rate[example_ids])
                    for layer_id in top_ids_lst
                ]
            else:
                base_rate_arr = [
                    layers[layer_id].base_rate[example_ids] for layer_id in top_ids_lst
                ]
            return np.concatenate(base_rate_arr)

    def set_top_mu(self, params, example_ids, valid, transform=True):
        layers = self.layers
        if valid:
            if transform:
                for count, layer_id in enumerate(self.top_ids_lst):
                    layers[layer_id].valid_base_rate[example_ids] = sp.fun(
                        params[
                            count * len(example_ids) : (count + 1) * len(example_ids)
                        ]
                    )
            else:
                for count, layer_id in enumerate(self.top_ids_lst):
                    layers[layer_id].valid_base_rate[example_ids] = params[
                        count * len(example_ids) : (count + 1) * len(example_ids)
                    ]
        else:
            if transform:
                for count, layer_id in enumerate(self.top_ids_lst):
                    layers[layer_id].base_rate[example_ids] = sp.fun(
                        params[
                            count * len(example_ids) : (count + 1) * len(example_ids)
                        ]
                    )
            else:
                for count, layer_id in enumerate(self.top_ids_lst):
                    layers[layer_id].base_rate[example_ids] = params[
                        count * len(example_ids) : (count + 1) * len(example_ids)
                    ]

    def get_top_virtual_mu(self, example_ids, valid, transform=True):
        layers = self.layers
        top_ids_lst = self.top_ids_lst
        if valid:
            if transform:
                base_rate_arr = [
                    sp.inv_fun(layers[layer_id].virtual_valid_base_rate[example_ids])
                    for layer_id in top_ids_lst
                ]
            else:
                base_rate_arr = [
                    layers[layer_id].virtual_valid_base_rate[example_ids]
                    for layer_id in top_ids_lst
                ]
            return np.concatenate(base_rate_arr)
        else:
            if transform:
                base_rate_arr = [
                    sp.inv_fun(layers[layer_id].virtual_base_rate[example_ids])
                    for layer_id in top_ids_lst
                ]
            else:
                base_rate_arr = [
                    layers[layer_id].virtual_base_rate[example_ids]
                    for layer_id in top_ids_lst
                ]
            return np.concatenate(base_rate_arr)

    def set_top_virtual_mu(self, params, example_ids, valid, transform=True):
        layers = self.layers
        if valid:
            if transform:
                for count, layer_id in enumerate(self.top_ids_lst):
                    layers[layer_id].virtual_valid_base_rate[example_ids] = sp.fun(
                        params[
                            count * len(example_ids) : (count + 1) * len(example_ids)
                        ]
                    )
            else:
                for count, layer_id in enumerate(self.top_ids_lst):
                    layers[layer_id].virtual_valid_base_rate[example_ids] = params[
                        count * len(example_ids) : (count + 1) * len(example_ids)
                    ]

        else:
            if transform:
                for count, layer_id in enumerate(self.top_ids_lst):
                    layers[layer_id].virtual_base_rate[example_ids] = sp.fun(
                        params[
                            count * len(example_ids) : (count + 1) * len(example_ids)
                        ]
                    )
            else:
                for count, layer_id in enumerate(self.top_ids_lst):
                    layers[layer_id].virtual_base_rate[example_ids] = params[
                        count * len(example_ids) : (count + 1) * len(example_ids)
                    ]

    def get_real_kernel_params(self, transform=True):
        params = []
        layers = self.layers
        real_kernel_layer_ids = [
            layer_id
            for layer_id in self.nontop_ids_lst
            if layer_id not in self.evidences_ids_set
        ]
        real_kernel_layer_ids = self.top_ids_lst + real_kernel_layer_ids

        for layer_id in real_kernel_layer_ids:
            this_layer = layers[layer_id]
            for kernel_param in this_layer.kernels_param_lst:
                for v, p in zip(this_layer.var_ids, kernel_param):
                    if transform:
                        params.append(sp.inv_fun(p))
                    else:
                        params.append(p)
        return params

    def set_real_kernel_params(self, layers_kernels_param_lst, transform=True):
        count_for_params = 0
        layers = self.layers
        real_kernel_layer_ids = [
            layer_id
            for layer_id in self.nontop_ids_lst
            if layer_id not in self.evidences_ids_set
        ]
        real_kernel_layer_ids = self.top_ids_lst + real_kernel_layer_ids
        for layer_id in real_kernel_layer_ids:
            this_layer = layers[layer_id]
            params_number = len(this_layer.var_ids) * len(this_layer.kernels_type_lst)
            params_to_set = layers_kernels_param_lst[
                count_for_params : count_for_params + params_number
            ]
            self.layers[layer_id].set_kernels_param(
                kernels_params_lst=params_to_set, transform=transform,
            )
            count_for_params += params_number

    def get_virtual_params(self, transform=True):
        params = []
        params_append = params.append
        layers = self.layers
        var_mu = False
        for layer_id in self.nontop_ids_lst:
            layer = layers[layer_id]
            var_ids = layer.virtual_var_ids
            if var_ids[0] == 0:
                var_ids_remaining = var_ids[1:]
                var_mu = True
            else:
                var_ids_remaining = var_ids
                var_mu = False
            if var_mu:
                if transform:
                    if layer.virtual_base_rate == 0:
                        params_append(-np_inf)
                    else:
                        params_append(sp.inv_fun(layer.virtual_base_rate))
                else:
                    params.append(layer.virtual_base_rate)
            for virtual_kernel_param in layer.virtual_kernels_param_lst:
                for v, p in zip(var_ids_remaining, virtual_kernel_param):
                    if transform:
                        params_append(sp.inv_fun(p))
                    else:
                        params_append(p)
        return params

    def set_virtual_params(self, layers_kernels_param_lst, transform=True):
        count_for_params = 0
        layers = self.layers

        var_mu = False
        for layer_id in self.nontop_ids_lst:
            this_layer = layers[layer_id]
            var_ids = this_layer.virtual_var_ids
            if var_ids[0] == 0:
                var_mu = True
            else:
                var_mu = False
            if var_mu:
                params_number = 1 + (len(var_ids) - 1) * len(
                    layers[layer_id].virtual_kernels_type_lst
                )
            else:
                params_number = len(var_ids) * len(
                    layers[layer_id].virtual_kernels_type_lst
                )
            params_to_set = layers_kernels_param_lst[
                count_for_params : count_for_params + params_number
            ]
            self.layers[layer_id].set_virtual_param(
                virtual_kernels_params_lst=params_to_set, transform=transform,
            )
            count_for_params += params_number


class DPPLayersEvents:
    """
    This is a class for doing sampling.
    """
    def __init__(
        self, dpp_layers, ex_id, valid, end_time, evidences_this_ex,
    ):
        self.layers = []
        for dpp_layer in dpp_layers.layers:
            if "virtual_kernels_param_lst" not in dpp_layer.__dict__:
                layer_to_append = TopLayerEvents(dpp_layer, ex_id, valid, end_time)
                self.layers.append(layer_to_append)
            else:
                layer_to_append = NonTopLayerEvents(dpp_layer, end_time)
                self.layers.append(layer_to_append)
        self.parents_ids_lst = dpp_layers.parents_ids_lst
        self.children_ids_lst = dpp_layers.children_ids_lst
        self.evidences_ids_set = dpp_layers.evidences_ids_set
        for key, value in evidences_this_ex.items():
            self.layers[key].real_events = value

        self.top_ids_lst = dpp_layers.top_ids_lst

    def set_layer_events(
        self,
        this_layer_id,
        real_events,
        virtual_events,
        this_layer_ll=None,
        this_layer_virtual_ll=None,
    ):
        layers = self.layers
        this_layer = layers[this_layer_id]
        if real_events is None:
            if virtual_events is not None:
                this_layer.virtual_events = virtual_events
        else:
            this_layer.real_events = real_events
            if virtual_events is not None:
                this_layer.virtual_events = virtual_events

        if this_layer_ll is None:
            parents_ids_keys = self.parents_ids_lst[this_layer_id].keys()
            if len(parents_ids_keys) == 0:
                this_layer.loglikelihood = this_layer.ll()
            else:
                upper_layers_lst = [layers[p_id] for p_id in parents_ids_keys]
                p_min_lst = [
                    np_min(layer.real_events) if len(layer.real_events) != 0 else np_inf
                    for layer in upper_layers_lst
                ]
                p_min = np_min(p_min_lst)

                stats_lst_ids_lst = [
                    self.children_ids_lst[u_id][this_layer_id]
                    for u_id in parents_ids_keys
                ]

                intensity_mat_lst = calc_intensity_mat_lst(
                    upper_layers_lst=upper_layers_lst,
                    events_times=this_layer.real_events,
                    stats_lst_ids_lst=stats_lst_ids_lst,
                )
                intensity_arr = calc_intensity_arr(
                    p_min=p_min,
                    intensity_mat_lst=intensity_mat_lst,
                    events_times=this_layer.real_events,
                    base_rate=this_layer.base_rate,
                )
                this_layer.intensity_arr = intensity_arr
                this_layer.parents_events_time_to_end_expec_lst = [
                    layer.kernels_lst[stats_lst_id].expectation(
                        input_arr=layer.end_time - layer.real_events, start_time=0
                    )
                    for layer, stats_lst_id in zip(upper_layers_lst, stats_lst_ids_lst)
                ]
                this_layer.intensity_mat_lst = intensity_mat_lst
                this_layer.loglikelihood = this_layer.ll(
                    this_layer.parents_events_time_to_end_expec_lst, intensity_arr
                )
        else:
            this_layer.loglikelihood = this_layer_ll

        if this_layer_virtual_ll is None and this_layer.virtual_events is not None:
            children_id_keys = self.children_ids_lst[this_layer_id].keys()
            lower_layers_lst = [layers[c_id] for c_id in children_id_keys]

            stats_lst_ids_lst = [
                self.parents_ids_lst[c_id][this_layer_id] for c_id in children_id_keys
            ]

            virtual_intensity_mat_lst = calc_virtual_intensity_mat_lst(
                lower_layers_lst=lower_layers_lst,
                events_times=this_layer.virtual_events,
                stats_lst_ids_lst=stats_lst_ids_lst,
            )
            this_layer.virtual_intensity_arr = calc_virtual_intensity_arr(
                virtual_intensity_mat_lst=virtual_intensity_mat_lst,
                virtual_events_times=this_layer.virtual_events,
                virtual_base_rate=this_layer.virtual_base_rate,
                lower_layers_lst=lower_layers_lst,
                stats_lst_ids_lst=stats_lst_ids_lst,
            )
            this_layer.virtual_intensity_mat_lst = virtual_intensity_mat_lst
            this_layer.children_events_time_to_beginning_expec_lst = [
                layer.virtual_kernels_lst[stats_lst_id].expectation(
                    input_arr=layer.real_events, start_time=0
                )
                for layer, stats_lst_id in zip(lower_layers_lst, stats_lst_ids_lst)
            ]
            this_layer.virtual_loglikelihood = this_layer.virtual_ll(
                children_events_time_to_beginning_expec_lst=this_layer.children_events_time_to_beginning_expec_lst,
                virtual_intensity_arr=this_layer.virtual_intensity_arr,
                lower_layers_lst=lower_layers_lst,
                stats_lst_ids_lst=stats_lst_ids_lst,
            )
        else:
            this_layer.virtual_loglikelihood = this_layer_virtual_ll


    def sample_first_event(self, rng):
        first_event_dict = {"type": None, "time": np_inf}  # e_layer_id, time
        forward_sampling_done_layers_set = set([])
        for e_layer_id in self.evidences_ids_set:
            self.sample_parents_first_event(
                e_layer_id, first_event_dict, forward_sampling_done_layers_set, rng
            )

        top_next_sample_time = None
        while first_event_dict["time"] == np_inf:
            top_layer_id_to_sample = self.top_ids_lst[0]
            top_layer_to_sample = self.layers[top_layer_id_to_sample]
            if top_next_sample_time is None:
                top_next_sample_time = (
                    rng.exponential(1 / top_layer_to_sample.base_rate)
                    + top_layer_to_sample.end_time
                )
            else:
                top_next_sample_time += rng.exponential(
                    1 / top_layer_to_sample.base_rate
                )
            assert top_next_sample_time != np_inf
            self.forward_sampling_to_the_evidence(
                top_layer_id_to_sample,
                first_event_dict,
                np_array([top_next_sample_time]),
                rng,
            )
        assert first_event_dict["time"] != np_inf
        for lst_id, top_layer_id in enumerate(self.top_ids_lst):
            this_layer = self.layers[top_layer_id]
            if lst_id == 0 and top_next_sample_time is not None:
                assert top_next_sample_time <= first_event_dict["time"]
                src_events = this_layer.homo_poisson_sample(
                    rng,
                    start_time=top_next_sample_time,
                    end_time=first_event_dict["time"],
                )
            else:
                assert this_layer.end_time <= first_event_dict["time"]
                src_events = this_layer.homo_poisson_sample(
                    rng,
                    start_time=this_layer.end_time,
                    end_time=first_event_dict["time"],
                )
            while len(src_events) > 0:
                self.forward_sampling_to_the_evidence(
                    top_layer_id, first_event_dict, src_events[0:10], rng
                )
                src_events = src_events[10:]
                src_events = src_events[src_events < first_event_dict["time"]]
        return first_event_dict

    def sample_parents_first_event(
        self, c_layer_id, first_event_dict, forward_sampling_done_layers_set, rng,
    ):
        parents_ids_keys = self.parents_ids_lst[c_layer_id].keys()
        for u_layer_id in parents_ids_keys:
            if u_layer_id not in forward_sampling_done_layers_set:
                forward_sampling_done_layers_set.update([u_layer_id])
                u_layer = self.layers[u_layer_id]
                src_events = u_layer.real_events
                if len(src_events) > 0:
                    self.forward_sampling_to_the_evidence(
                        u_layer_id, first_event_dict, src_events, rng
                    )
            self.sample_parents_first_event(
                u_layer_id, first_event_dict, forward_sampling_done_layers_set, rng
            )

    def forward_sampling_to_the_evidence(
        self, layer_id, first_event_dict, src_events, rng
    ):
        if self.children_ids_lst[layer_id]:
            for c_layer_id, stats_lst_id in self.children_ids_lst[layer_id].items():
                this_layer = self.layers[layer_id]
                mu = rng.exponential(scale=1, size=len(src_events))
                kernel = this_layer.kernels_lst[stats_lst_id]
                input_arr = np_where(
                    src_events > this_layer.end_time, src_events, this_layer.end_time
                )
                src_events_c_layer = kernel.expectation_inv(
                    mu + kernel.expectation(input_arr, src_events), src_events
                )
                if c_layer_id in self.evidences_ids_set:
                    potential_first_event = np_min(src_events_c_layer)
                    if potential_first_event < first_event_dict["time"]:
                        first_event_dict["time"] = potential_first_event
                        first_event_dict["type"] = c_layer_id
                    continue

                src_events_c_layer = src_events_c_layer[
                    src_events_c_layer < first_event_dict["time"]
                ]
                if len(src_events_c_layer) > 0:
                    self.forward_sampling_to_the_evidence(
                        c_layer_id, first_event_dict, src_events_c_layer, rng
                    )


def calc_virtual_intensity_mat_lst(lower_layers_lst, events_times, stats_lst_ids_lst):
    virtual_intensity_mat_lst = []
    virtual_intensity_mat_lst_append = virtual_intensity_mat_lst.append
    for lower_layer, stats_lst_id in zip(lower_layers_lst, stats_lst_ids_lst):

        lower_layer_sub_this_layer = lower_layer.real_events[:, None] - events_times
        empty_ids = lower_layer_sub_this_layer <= 0
        lower_layer_sub_this_layer[empty_ids] = np_inf
        virtual_intensity_mat = lower_layer.virtual_kernels_lst[stats_lst_id].fun(
            lower_layer_sub_this_layer, 0
        )
        virtual_intensity_mat_lst_append(virtual_intensity_mat)
    return virtual_intensity_mat_lst


def calc_virtual_intensity_arr(
    virtual_intensity_mat_lst,
    virtual_events_times,
    virtual_base_rate,
    lower_layers_lst,
    stats_lst_ids_lst,
):
    virtual_intensity_arr = [
        np_sum(virtual_intensity_mat, axis=0)
        for virtual_intensity_mat in virtual_intensity_mat_lst
    ]
    virtual_intensity_arr = np_sum(virtual_intensity_arr, axis=0)
    virtual_intensity_arr += virtual_base_rate

    end_time = lower_layers_lst[0].end_time
    for stats_lst_id, lower_layer in zip(stats_lst_ids_lst, lower_layers_lst):
        if lower_layer.synthetic_event_end:
            if len(lower_layer.real_events) == 0:
                end_time_intensity_mat = lower_layer.virtual_kernels_lst[stats_lst_id].fun(
                    end_time - virtual_events_times, 0
                )
                virtual_intensity_arr += end_time_intensity_mat
            elif np_max(lower_layer.real_events) < lower_layer.end_time:
                end_time_intensity_mat = lower_layer.virtual_kernels_lst[stats_lst_id].fun(
                    end_time - virtual_events_times, 0
                )
                virtual_intensity_arr += end_time_intensity_mat
    virtual_intensity_arr[virtual_intensity_arr < 1e-100] = 1e-100
    return virtual_intensity_arr


def calc_intensity_mat_lst(upper_layers_lst, events_times, stats_lst_ids_lst):
    intensity_mat_lst = []
    intensity_mat_lst_append = intensity_mat_lst.append
    for upper_layer, stats_lst_id in zip(upper_layers_lst, stats_lst_ids_lst):
        this_layer_sub_upper_layer = events_times[:, None] - upper_layer.real_events
        empty_ids = this_layer_sub_upper_layer <= 0
        this_layer_sub_upper_layer[empty_ids] = np_inf
        intensity_mat = upper_layer.kernels_lst[stats_lst_id].fun(
            this_layer_sub_upper_layer, 0
        )
        intensity_mat_lst_append(intensity_mat)
    return intensity_mat_lst


def calc_intensity_arr(p_min, intensity_mat_lst, events_times, base_rate):
    if not intensity_mat_lst and not hasattr(events_times, "__len__"):
        return np_array([base_rate])
    nonzero_entry = events_times > p_min
    intensity_arr = [
        np_sum(intensity_mat, axis=1) for intensity_mat in intensity_mat_lst
    ]
    intensity_arr = np_sum(intensity_arr, axis=0) + base_rate
    intensity_arr[np_logical_and(nonzero_entry, intensity_arr < 1e-100)] = 1e-100
    return intensity_arr


def sample_from_lower_layers_to_upper(
    lower_layers_lst,
    base_rate,
    rng,
    real_prior,
    start_time,
    end_time,
    stats_lst_ids_lst,
):
    virtual_events_deque = deque()

    true_base_rate = base_rate
    for lower_layer, stats_lst_id in zip(lower_layers_lst, stats_lst_ids_lst):
        single_virtual_events = lower_layer.nonhomo_poisson_inversion_reverse_to_upper(
            upper_layer_base_rate=base_rate,
            rng=rng,
            real_prior=real_prior,
            start_time=start_time,
            end_time=end_time,
            stats_lst_id=stats_lst_id,
        )
        virtual_events_deque.append(single_virtual_events)
        base_rate = 0

    base_rate = true_base_rate

    children_events_time_to_beginning_expec_lst = [
        lower_layer.virtual_kernels_lst[stats_lst_id].expectation(
            input_arr=lower_layer.real_events, start_time=0
        )
        for lower_layer, stats_lst_id in zip(lower_layers_lst, stats_lst_ids_lst)
    ]

    virtual_events = np_concatenate(virtual_events_deque)
    virtual_intensity_mat_lst = calc_virtual_intensity_mat_lst(
        lower_layers_lst=lower_layers_lst,
        events_times=virtual_events,
        stats_lst_ids_lst=stats_lst_ids_lst,
    )
    virtual_intensity_arr = calc_virtual_intensity_arr(
        virtual_intensity_mat_lst=virtual_intensity_mat_lst,
        virtual_events_times=virtual_events,
        virtual_base_rate=base_rate,
        lower_layers_lst=lower_layers_lst,
        stats_lst_ids_lst=stats_lst_ids_lst,
    )
    return (
        virtual_events,
        children_events_time_to_beginning_expec_lst,
        virtual_intensity_mat_lst,
        virtual_intensity_arr,
    )


def generate_kernels_lst(kernels_param_lst, kernels_type_lst):
    return [
        ExpKernel1(kernel_param[0], kernel_param[1])
        if kernel_type == "ExpKernel1"
        else ExpKernel2(kernel_param[0], kernel_param[1])
        if kernel_type == "ExpKernel2"
        else GammaKernel(kernel_param[0], kernel_param[1], kernel_param[2])
        if kernel_type == "GammaKernel"
        else PowerLawKernel(kernel_param[0], kernel_param[1], kernel_param[2])
        if kernel_type == "PowerLawKernel"
        else GammaKernelWithShift(
            kernel_param[0], kernel_param[1], kernel_param[2], kernel_param[3]
        )
        if kernel_type == "GammaKernelWithShift"
        else None
        for kernel_param, kernel_type in zip(kernels_param_lst, kernels_type_lst)
    ]


class DPPLayer:
    def __init__(
        self, base_rate, virtual_base_rate, var_ids, virtual_var_ids, layer_id,
    ):
        self.base_rate = base_rate
        self.virtual_base_rate = virtual_base_rate
        self.var_ids = var_ids
        self.virtual_var_ids = virtual_var_ids
        self.layer_id = layer_id
        self.kernels_type_lst = None
        self.kernels_param_lst = []
        self.kernels_lst = None

    def set_kernels_param(self, kernels_params_lst, transform):
        count_for_params = 0
        var_ids = self.var_ids
        for count in range(len(self.kernels_type_lst)):
            param = kernels_params_lst[
                count_for_params : count_for_params + len(var_ids)
            ]
            if transform:
                param = sp.fun(param)
            for i, p in zip(var_ids, param):
                p = 1e-10 if p < 1e-10 else p
                self.kernels_param_lst[count][i - 1] = p
            count_for_params += len(var_ids)
        self.kernels_lst = generate_kernels_lst(
            kernels_param_lst=self.kernels_param_lst,
            kernels_type_lst=self.kernels_type_lst,
        )


class DPPLayerEvents:
    def __init__(self, base_rate, virtual_base_rate, end_time=None, layer_id=None):
        self.base_rate = base_rate
        self.virtual_base_rate = virtual_base_rate
        self.layer_id = layer_id
        self.loglikelihood = -np_inf
        self.virtual_loglikelihood = -np_inf
        self.children_events_time_to_beginning_expec_lst = None
        self.virtual_intensity_mat_lst = None
        self.virtual_intensity_arr = None
        self.real_events = None
        self.virtual_events = None
        self.real_events_num = 0
        self.virtual_events_num = 0
        self.end_time = end_time
        self.kernels_lst = []

    def nonhomo_poisson_inversion(
        self, stat_lst_id, upper_layer_events_times, base_rate, rng, start_time=None, end_time=None
    ):
        start_time = 0 if start_time is None else start_time
        end_time = self.end_time if end_time is None else end_time
        events = deque()
        event_time = start_time
        kernel_expectation_inv = self.kernels_lst[stat_lst_id].expectation_inv
        kernel_ecpectation = self.kernels_lst[stat_lst_id].expectation
        rng_exponential = rng.exponential
        if base_rate > 1e-10:
            while True:
                delta = rng_exponential(1 / base_rate)
                event_time += delta
                if event_time > end_time:
                    break
                events.append(event_time)

        for t_0 in upper_layer_events_times:
            t = t_0 if t_0 > start_time else start_time
            while True:
                mu = rng_exponential(1)
                t = kernel_expectation_inv(mu + kernel_ecpectation(t, t_0), t_0)
                if t > end_time:
                    break
                events.append(t)
        return np_array(events)

    def virtual_ll(
        self,
        children_events_time_to_beginning_expec_lst,
        virtual_intensity_arr,
        lower_layers_lst,
        stats_lst_ids_lst,
    ):
        if children_events_time_to_beginning_expec_lst is None:
            children_events_time_to_beginning_expec_lst = (
                self.children_events_time_to_beginning_expec_lst
            )
        if virtual_intensity_arr is None:
            virtual_intensity_arr = self.virtual_intensity_arr

        ll_time = -np_sum(
            [np_sum(expec) for expec in children_events_time_to_beginning_expec_lst]
        )
        end_time = self.end_time
        ll_time -= end_time * self.virtual_base_rate
        for lower_layer, stats_lst_id in zip(lower_layers_lst, stats_lst_ids_lst):
            if lower_layer.synthetic_event_end:
                if len(lower_layer.real_events) == 0: 
                    ll_time -= lower_layer.virtual_kernels_lst[stats_lst_id].expectation(
                        input_arr=end_time, start_time=0
                    )
                elif np_max(lower_layer.real_events) < lower_layer.end_time:
                    ll_time -= lower_layer.virtual_kernels_lst[stats_lst_id].expectation(
                        input_arr=end_time, start_time=0
                    )
        ll_events = np_sum(np_log(virtual_intensity_arr))
        ll = ll_time + ll_events
        return ll

    def prior_sample(
        self,
        lower_layers_lst,
        virtual=False,
        rng=None,
        start_time=None,
        end_time=None,
        stats_lst_ids_lst=None,
    ):
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = self.end_time

        if not "virtual_kernels_type_lst" in self.__dict__ and self.base_rate == 0:
            self.real_events = np_array([])
        else:
            lower_layers_real_events = [
                layer.real_events
                for layer in lower_layers_lst
                if len(layer.real_events) != 0
            ]
            if lower_layers_real_events:
                c_min = np_min(
                    [np_min(real_events) for real_events in lower_layers_real_events]
                )
            while True:
                (self.real_events, _, _, _,) = sample_from_lower_layers_to_upper(
                    lower_layers_lst=lower_layers_lst,
                    base_rate=self.virtual_base_rate,
                    rng=rng,
                    real_prior=True,
                    start_time=start_time,
                    end_time=end_time,
                    stats_lst_ids_lst=stats_lst_ids_lst,
                )
                if lower_layers_real_events:
                    if "virtual_kernels_type_lst" in self.__dict__ or (
                        not "virtual_kernels_type_lst" in self.__dict__
                        and self.base_rate != 0
                    ):
                        if (
                            len(self.real_events) != 0
                            and np_min(self.real_events) < c_min
                        ):
                            break
                else:
                    break
        if virtual:
            (
                self.virtual_events,
                self.children_events_time_to_beginning_expec_lst,
                self.virtual_intensity_mat_lst,
                self.virtual_intensity_arr,
            ) = sample_from_lower_layers_to_upper(
                lower_layers_lst=lower_layers_lst,
                base_rate=self.virtual_base_rate,
                rng=rng,
                real_prior=False,
                start_time=start_time,
                end_time=end_time,
                stats_lst_ids_lst=stats_lst_ids_lst,
            )
            self.virtual_loglikelihood = self.virtual_ll(
                children_events_time_to_beginning_expec_lst=self.children_events_time_to_beginning_expec_lst,
                virtual_intensity_arr=self.virtual_intensity_arr,
                lower_layers_lst=lower_layers_lst,
                stats_lst_ids_lst=stats_lst_ids_lst,
            )


class TopLayer(DPPLayer):
    def __init__(
        self,
        base_rate,
        virtual_base_rate,
        valid_base_rate,
        virtual_valid_base_rate,
        kernels_type_lst,
        kernels_param_lst,
        var_ids,
        virtual_var_ids,
        layer_id,
    ):
        DPPLayer.__init__(
            self, base_rate, virtual_base_rate, var_ids, virtual_var_ids, layer_id,
        )
        self.valid_base_rate = valid_base_rate
        self.virtual_valid_base_rate = virtual_valid_base_rate
        self.kernels_type_lst = kernels_type_lst
        self.kernels_param_lst = kernels_param_lst
        self.kernels_lst = generate_kernels_lst(
            kernels_param_lst=kernels_param_lst, kernels_type_lst=kernels_type_lst
        )


class TopLayerEvents(DPPLayerEvents):
    def __init__(self, top_layer, ex_id, valid, end_time):
        DPPLayerEvents.__init__(
            self, base_rate=None, virtual_base_rate=None,
        )
        if not valid:
            self.base_rate = top_layer.base_rate[
                ex_id
            ]
            self.virtual_base_rate = top_layer.virtual_base_rate[ex_id]
        else:
            self.base_rate = top_layer.valid_base_rate[ex_id]
            self.virtual_base_rate = top_layer.virtual_valid_base_rate[ex_id]
        self.kernels_type_lst = top_layer.kernels_type_lst
        self.kernels_param_lst = top_layer.kernels_param_lst
        self.kernels_lst = top_layer.kernels_lst
        self.layer_id = top_layer.layer_id
        self.end_time = end_time

    def homo_poisson_sample(self, rng, start_time=None, end_time=None):
        start_time = 0 if start_time is None else start_time
        end_time = self.end_time if end_time is None else end_time
        base_rate = self.base_rate
        if self.base_rate > 1e-10:
            poisson_num = rng.poisson(lam=base_rate * (end_time - start_time))
            events = rng.uniform(low=start_time, high=end_time, size=poisson_num)
            events = np_sort(events)
        else:
            events = np_array([])
        return events

    def layer_expectation(self):
        return self.end_time * self.base_rate

    def ll(self, events_times=None):
        if events_times is None:
            events_times = self.real_events
        if len(events_times) == 0:
            return -self.layer_expectation()
        elif self.base_rate == 0:
            return -np_inf
        loglikelihood_val = -self.layer_expectation() + len(events_times) * np_log(
            self.base_rate
        )
        return loglikelihood_val


class NonTopLayer(DPPLayer):
    def __init__(
        self,
        base_rate,
        virtual_base_rate,
        kernels_type_lst,
        virtual_kernels_type_lst,
        kernels_param_lst,
        virtual_kernels_param_lst,
        var_ids,
        virtual_var_ids,
        layer_id,
        synthetic_event_end,
    ):
        DPPLayer.__init__(
            self, base_rate, virtual_base_rate, var_ids, virtual_var_ids, layer_id
        )
        self.kernels_param_lst = kernels_param_lst
        self.virtual_kernels_param_lst = virtual_kernels_param_lst
        self.kernels_type_lst = kernels_type_lst
        self.virtual_kernels_type_lst = virtual_kernels_type_lst
        self.kernels_lst = generate_kernels_lst(
            kernels_param_lst=kernels_param_lst, kernels_type_lst=kernels_type_lst
        )
        self.virtual_kernels_lst = generate_kernels_lst(
            kernels_param_lst=virtual_kernels_param_lst,
            kernels_type_lst=virtual_kernels_type_lst,
        )
        self.synthetic_event_end = synthetic_event_end

    def set_virtual_param(self, virtual_kernels_params_lst, transform):
        count_for_params = 0
        if transform:
            virtual_kernels_params_lst = sp.fun(virtual_kernels_params_lst)
        var_ids = self.virtual_var_ids
        if var_ids[0] == 0:
            self.virtual_base_rate = virtual_kernels_params_lst[0]
            virtual_kernels_params_lst = virtual_kernels_params_lst[1:]
            var_ids_remaining = var_ids[1:]
        else:
            var_ids_remaining = var_ids

        for count in range(len(self.virtual_kernels_type_lst)):
            param = virtual_kernels_params_lst[
                count_for_params : count_for_params + len(var_ids_remaining)
            ]
            for i, p in zip(var_ids_remaining, param):
                p = 1e-10 if p < 1e-10 else p
                self.virtual_kernels_param_lst[count][i - 1] = p
            count_for_params += len(var_ids_remaining)
        self.virtual_kernels_lst = generate_kernels_lst(
            kernels_param_lst=self.virtual_kernels_param_lst,
            kernels_type_lst=self.virtual_kernels_type_lst,
        )


class NonTopLayerEvents(DPPLayerEvents):
    def __init__(self, nontop_layer, end_time):
        DPPLayerEvents.__init__(
            self,
            base_rate=nontop_layer.base_rate,
            virtual_base_rate=nontop_layer.virtual_base_rate,
        )
        self.kernels_type_lst = nontop_layer.kernels_type_lst
        self.virtual_kernels_type_lst = nontop_layer.virtual_kernels_type_lst

        self.kernels_param_lst = nontop_layer.kernels_param_lst
        self.virtual_kernels_param_lst = nontop_layer.virtual_kernels_param_lst

        self.kernels_lst = nontop_layer.kernels_lst
        self.virtual_kernels_lst = nontop_layer.virtual_kernels_lst

        self.parents_events_time_to_end_expec_lst = None
        self.intensity_mat_lst = None
        self.intnsity_arr = None
        self.synthetic_event_end = nontop_layer.synthetic_event_end
        self.end_time = end_time
        self.layer_id = nontop_layer.layer_id


    def nonhomo_poisson_inversion_reverse_to_upper(
        self,
        upper_layer_base_rate,
        rng,
        real_prior,
        start_time=None,
        end_time=None,
        stats_lst_id=None,
    ):
        start_time = 0 if start_time is None else start_time
        end_time = self.end_time if end_time is None else end_time
        events = deque()
        events_append = events.append
        event_time = start_time
        virtual_kernel_expectation_inv = self.virtual_kernels_lst[
            stats_lst_id
        ].expectation_inv
        virtual_kernel_expectation = self.virtual_kernels_lst[stats_lst_id].expectation
        rng_exponential = rng.exponential
        if upper_layer_base_rate > 1e-10:
            while True:
                delta = rng_exponential(1 / upper_layer_base_rate)
                event_time += delta
                if event_time >= end_time:
                    break
                events_append(event_time)

        if self.synthetic_event_end:
            if len(self.real_events) == 0:
                real_events = np_append(self.real_events, self.end_time)
            elif np.max(self.real_events) < self.end_time:
                real_events = np_append(self.real_events, self.end_time)
            else:
                real_events = self.real_events
        else:
            real_events = self.real_events
        for t_0 in real_events:
            t = t_0 if t_0 < end_time else end_time
            while True:
                mu = rng_exponential(1)
                t = virtual_kernel_expectation_inv(
                    mu + virtual_kernel_expectation(t, t_0), t_0
                )
                if t > 2 * t_0 - start_time:
                    break
                if t != t_0:
                    events_append(2 * t_0 - t)
                    if real_prior:
                        break

        return np_array(events)

    def ll(
        self, parents_events_time_to_end_expec_lst, intensity_arr,
    ):
        if parents_events_time_to_end_expec_lst is None:
            parents_events_time_to_end_expec_lst = (
                self.parents_events_time_to_end_expec_lst
            )

        if np.any(intensity_arr == 0):
            return -np_inf
        ll_time = -np_sum(
            [np_sum(expec) for expec in parents_events_time_to_end_expec_lst]
        )
        ll_time -= self.base_rate * self.end_time
        ll_events = np_sum(np_log(intensity_arr))
        ll = ll_time + ll_events
        return ll


class ExpKernel1:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def fun(self, input_arr, start_time):
        return self.alpha * np_exp(-self.beta * (input_arr - start_time))

    def expectation(self, input_arr, start_time, int_start_time=None):
        if int_start_time is None or int_start_time < 1e-6:
            int_mult = 1.0
            int_start_time = start_time
        else:
            int_start_time_bool = int_start_time > start_time
            int_mult = np_where(
                int_start_time_bool,
                np_exp(-self.beta * (int_start_time - start_time)),
                np.ones(start_time.shape),
            )
            int_start_time = np_where(int_start_time_bool, int_start_time, start_time)
        # return self.alpha * (1 - np_exp(-self.beta * (input_arr - start_time))) / self.beta
        return (
            -self.alpha
            * int_mult
            * np.expm1(-self.beta * (input_arr - int_start_time))
            / self.beta
        )

    def expectation_inv(self, input_arr, start_time):
        if np.any(np.array([1 - self.beta * input_arr / self.alpha]) < 0):
            return np_inf
        # return start_time - np.log(1 - self.beta * input_arr / self.alpha) / self.beta
        return start_time - np_log1p(-self.beta * input_arr / self.alpha) / self.beta


class ExpKernel2:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def fun(self, input_arr, start_time):
        return self.alpha * self.beta * np_exp(-self.beta * (input_arr - start_time))

    def expectation(self, input_arr, start_time, int_start_time=None):
        if int_start_time is None or int_start_time < 1e-6:
            int_mult = 1.0
            int_start_time = start_time
        else:
            int_start_time_bool = int_start_time > start_time
            diff_int_start_n_start = np_where(
                int_start_time_bool, int_start_time - start_time, 0
            )
            # int_mult = np_where(
            #     int_start_time_bool,
            #     np.exp(-self.beta * (int_start_time - start_time)),
            #     np.ones(start_time.shape),
            # )
            int_mult = np_exp(-self.beta * diff_int_start_n_start)
            int_start_time = np_where(int_start_time_bool, int_start_time, start_time)
        return (
            -self.alpha
            * int_mult
            * (np_exp(-self.beta * (input_arr - int_start_time)) - 1)
        )

    def expectation_inv(self, input_arr, start_time):
        # if np.any(np.array([1 - input_arr / self.alpha]) < 0):
        #     return np_inf
        if (1 - input_arr / self.alpha) <= 0:
            return np_inf
        return start_time - np_log1p(-input_arr / self.alpha) / self.beta


class GammaKernel:
    def __init__(self, partition, shape, rate):
        self.partition = partition
        self.shape = shape
        self.rate = rate
        self.gamma_shape = gamma(shape)

    def fun(self, input_arr, start_time):
        input_arr_copy = copy.deepcopy(input_arr)
        input_arr_copy -= start_time
        fake_input_arr = copy.deepcopy(input_arr_copy)
        if hasattr(fake_input_arr, "__len__"):
            fake_input_arr[fake_input_arr == np_inf] = 1
        elif fake_input_arr == np_inf:
            fake_input_arr = 1
        return (
            self.partition
            * self.rate ** self.shape
            / self.gamma_shape
            * fake_input_arr ** (self.shape - 1)
            * np_exp(-self.rate * input_arr_copy)
        )

    def expectation(self, input_arr, start_time, int_start_time=None):
        fake_input_arr = input_arr - start_time
        if int_start_time is None or int_start_time < 1e-6:
            return self.partition * gammainc(self.shape, self.rate * fake_input_arr)
        else:
            int_start_time_bool = int_start_time > start_time
            int_start_time = np_where(int_start_time_bool, int_start_time, start_time)
            int_start_time -= start_time
            return self.partition * (
                gammainc(self.shape, self.rate * fake_input_arr)
                - gammainc(self.shape, self.rate * int_start_time)
            )

    def expectation_inv(self, input_arr, start_time):
        if hasattr(input_arr, "__len__"):
            return np_where(
                input_arr >= self.partition,
                np_inf,
                start_time
                + gammaincinv(self.shape, input_arr / self.partition) / self.rate,
            )
        else:
            if input_arr / self.partition >= 1:
                return np_inf
            return (
                start_time
                + gammaincinv(self.shape, input_arr / self.partition) / self.rate
            )


class PowerLawKernel:
    def __init__(self, alpha, beta, c):
        self.alpha = alpha
        self.beta = beta
        self.c = c

    def fun(self, input_arr, start_time):
        input_arr -= start_time
        return (
            self.alpha
            * self.beta
            * self.c ** self.beta
            / (input_arr + self.c) ** (1 + self.beta)
        )

    def expectation(self, input_arr, start_time, int_start_time=None):
        input_arr -= start_time
        if int_start_time is None or int_start_time < 1e-6:
            return (
                self.alpha
                * self.c ** self.beta
                * (-((input_arr + self.c) ** (-self.beta)) + self.c ** (-self.beta))
            )
        else:
            int_start_time_bool = int_start_time > start_time
            int_start_time = np_where(int_start_time_bool, int_start_time, start_time)
            int_start_time -= start_time
            return (
                self.alpha
                * self.c ** self.beta
                * (
                    (int_start_time + self.c) ** (-self.beta)
                    - (input_arr + self.c) ** (-self.beta)
                )
            )

    def expectation_inv(self, input_arr, start_time):
        if input_arr >= self.alpha:
            return np_inf
        else:
            return start_time + (
                -input_arr / (self.alpha * self.c ** self.beta) + self.c ** (-self.beta)
            ) ** (-1 / self.beta)


class GammaKernelWithShift:
    def __init__(self, partition, shape, rate, c):
        self.partition = partition
        self.shape = shape
        self.rate = rate
        self.c = c
        self.gammaincc_const = gammaincc(self.shape, self.rate * self.c)
        self.gammainc_const = gammainc(self.shape, self.rate * self.c)
        self.gammaincc_times_gammashape = self.gammaincc_const * gamma(shape)

    def fun(self, input_arr, start_time):
        input_arr -= start_time
        fake_input_arr = copy.deepcopy(input_arr)
        if hasattr(fake_input_arr, "__len__"):
            fake_input_arr[fake_input_arr == np_inf] = 1
        elif fake_input_arr == np_inf:
            fake_input_arr = 1
        return (
            self.partition
            * self.rate ** self.shape
            / self.gammaincc_times_gammashape
            * (fake_input_arr + self.c) ** (self.shape - 1)
            * np_exp(-self.rate * (input_arr + self.c))
        )

    def expectation(self, input_arr, start_time, int_start_time=None):
        input_arr -= start_time
        if int_start_time is None or int_start_time < 1e-6:
            return (
                self.partition
                / self.gammaincc_const
                * (
                    gammainc(self.shape, self.rate * (input_arr + self.c))
                    - self.gammainc_const
                )
            )
        else:
            int_start_time_bool = int_start_time > start_time
            int_start_time = np_where(int_start_time_bool, int_start_time, start_time)
            int_start_time -= start_time
            return (
                self.partition
                / self.gammaincc_const
                * (
                    gammainc(self.shape, self.rate * (input_arr + self.c))
                    - gammainc(self.shape, self.rate * (int_start_time + self.c))
                )
            )

    def expectation_inv(self, input_arr, start_time):
        if input_arr / self.partition >= 1:
            return np_inf
        return (
            start_time
            + gammaincinv(
                self.shape,
                input_arr / self.partition * self.gammaincc_const + self.gammainc_const,
            )
            / self.rate
            - self.c
        )

