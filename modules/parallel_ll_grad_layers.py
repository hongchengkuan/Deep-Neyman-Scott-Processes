import numpy as np
from multiprocessing import Pool
from collections import deque, namedtuple
from DPPLayers import sp
from scipy.special import gamma, gammainc, digamma, gammaincc
import copy


np_sum = np.sum
np_array = np.array
np_log = np.log
np_min = np.min
np_inf = np.inf
np_logical_and = np.logical_and
np_exp = np.exp
np_append = np.append
np_mean = np.mean
np_logical_not = np.logical_not
np_concatenate = np.concatenate
np_reshape = np.reshape
np_zeros = np.zeros
np_frompyfunc = np.frompyfunc
np_float64 = np.float64
np_any = np.any

layers_samples = {}
virtual_layers_samples = {}
end_time_tuple = ()
layers = None
top_ids_lst = None
nontop_ids_lst = None
valid = None
num_of_mcmc_samples = 0
jac = None
virtual_jac = None
virtual_wrt_real_jac = None
maximize = None
children_ids_lst = None
parents_ids_lst = None
evidences_dict = None
evidences_ids_set = None

Stats_parallel_collections = namedtuple(
    "Stats_parallel_collections",
    [
        "top_ll_lst",
        "top_mu_grad_lst",
        "top_base_rate_lst",
        "ll_dq",
        "grad_dq",
        "virtual_top_ll_dq",
        "virtual_top_ll_wrt_real_dq",
        "virtual_top_grad_mu_dq",
        "virtual_top_grad_mu_wrt_real_dq",
        "virtual_ll_dq",
        "virtual_ll_wrt_real_dq",
        "virtual_grad_arr",
        "virtual_grad_wrt_real_arr",
    ],
)
Expkernel2gradstats = namedtuple(
    "Expkernel2gradstats",
    [
        "parents_events_time_to_end_expec_lst",
        "intensity_array",
        "parents_events_time_to_end_lst",
        "events_intervals_lst",
        "exp_intervals_div_intensity_lst",
    ],
)

Powerlawkernelgradstats = namedtuple(
    "Powerlawkernelgradstats",
    [
        "parents_events_time_to_end_expec_lst",
        "intensity_array",
        "parents_events_time_to_end_lst",
        "events_intervals_lst",
        "power_intervals_div_intensity_lst",
    ],
)

Llstats = namedtuple(
    "Llstats", ["parents_events_time_to_end_expec_lst", "intensity_array",]
)

Powerlawkernelgrad_virtual_stats = namedtuple(
    "Powerlawkernelgrad_virtual_stats",
    [
        "children_events_time_to_beginning_expec_with_syn_end_lst",
        "virtual_intensity_array",
        "children_events_time_to_beginning_with_syn_end_lst",
        "events_intervals_lst",
        "power_intervals_div_intensity_lst",
    ],
)

Expkernel2grad_virtual_stats = namedtuple(
    "Expkernel2grad_virtual_stats",
    [
        "children_events_time_to_beginning_expec_with_syn_end_lst",
        "virtual_intensity_array",
        "children_events_time_to_beginning_with_syn_end_lst",
        "events_intervals_lst",
        "exp_intervals_div_intensity_lst",
    ],
)

Ll_virtual_stats = namedtuple(
    "Ll_virtual_stats",
    [
        "children_events_time_to_beginning_expec_with_syn_end_lst",
        "virtual_intensity_array",
    ],
)


def initialize_global_vars(
    dpp_layers_samples,
    jac_init,
    virtual_jac_init,
    virtual_wrt_real_jac_init,
    maximize_init,
    evidences_dict_init,
):
    global layers_samples
    global virtual_layers_samples
    global end_time_tuple
    global layers
    global top_ids_lst
    global nontop_ids_lst
    global valid
    global num_of_mcmc_samples
    global jac
    global virtual_jac
    global virtual_wrt_real_jac
    global maximize
    global parents_ids_lst
    global children_ids_lst
    global evidences_dict
    global evidences_ids_set
    dpp_layers = dpp_layers_samples.dpp_layers
    layers_samples = dpp_layers_samples.layers_samples
    virtual_layers_samples = dpp_layers_samples.layers_virtual_samples
    end_time_tuple = dpp_layers_samples.end_time_tuple
    layers = dpp_layers.layers
    top_ids_lst = dpp_layers.top_ids_lst
    nontop_ids_lst = dpp_layers.nontop_ids_lst
    valid = dpp_layers_samples.valid
    num_of_mcmc_samples = dpp_layers_samples.num_of_mcmc_samples
    jac = jac_init
    virtual_jac = virtual_jac_init
    virtual_wrt_real_jac = virtual_wrt_real_jac_init
    maximize = maximize_init
    parents_ids_lst = dpp_layers.parents_ids_lst
    children_ids_lst = dpp_layers.children_ids_lst
    evidences_dict = evidences_dict_init
    evidences_ids_set = dpp_layers.evidences_ids_set


def ll_grad_for_all_layers(ex_id):
    top_ll_lst = None
    top_mu_grad_lst = None
    top_base_rate_lst = None
    ll_dq = None
    grad_dq = None
    virtual_top_ll_dq = None
    virtual_top_ll_wrt_real_dq = None
    virtual_top_grad_mu_dq = None
    virtual_top_grad_mu_wrt_real_dq = None
    virtual_ll_dq = None
    virtual_ll_wrt_real_dq = None
    virtual_grad_arr = None
    virtual_grad_wrt_real_arr = None

    if jac and maximize:
        top_ll_lst, top_mu_grad_lst, top_base_rate_lst = ll_grad_for_top_layers(
            top_ids_lst=top_ids_lst, ex_id=ex_id, jac_bool=jac,
        )
    elif jac:
        top_ll_lst, top_mu_grad_lst = ll_grad_for_top_layers(
            top_ids_lst=top_ids_lst, ex_id=ex_id, jac_bool=jac
        )
    elif maximize:
        top_ll_lst, top_base_rate_lst = ll_grad_for_top_layers(
            top_ids_lst=top_ids_lst, ex_id=ex_id, jac_bool=jac
        )
    else:
        top_ll_lst = ll_grad_for_top_layers(
            top_ids_lst=top_ids_lst, ex_id=ex_id, jac_bool=jac
        )

    this_example_samples = layers_samples[ex_id]
    this_example_virtual_samples = virtual_layers_samples[ex_id]
    l_children_ids_lst = children_ids_lst
    if virtual_jac:
        (
            virtual_top_ll_dq,
            virtual_top_grad_mu_dq,
            virtual_top_grad_dq,
        ) = virtual_ll_grad_for_top_layers(
            top_ids_lst=top_ids_lst,
            ex_id=ex_id,
            this_example_samples=this_example_samples,
            this_example_virtual_samples=this_example_virtual_samples,
            children_ids_lst=l_children_ids_lst,
            jac_bool=virtual_jac,
        )
        (
            virtual_ll_dq,
            virtual_grad_dq,
            virtual_grad_mu_dq,
        ) = virtual_ll_grad_for_nontop_layers(
            nontop_ids_lst=nontop_ids_lst,
            ex_id=ex_id,
            this_example_samples=this_example_samples,
            this_example_virtual_samples=this_example_virtual_samples,
            children_ids_lst=l_children_ids_lst,
            jac_bool=virtual_jac,
        )
        for virtual_grad_mu_dq_sample in virtual_grad_mu_dq:
            virtual_grad_mu_dq_sample += deque([0] * len(evidences_ids_set))
        virtual_grad_arr = organize_virtual_grad(
            virtual_top_grad_dq=virtual_top_grad_dq,
            virtual_grad_dq=virtual_grad_dq,
            virtual_grad_mu_dq=virtual_grad_mu_dq,
        )
    else:
        virtual_top_ll_dq = virtual_ll_grad_for_top_layers(
            top_ids_lst=top_ids_lst,
            ex_id=ex_id,
            this_example_samples=this_example_samples,
            this_example_virtual_samples=this_example_virtual_samples,
            children_ids_lst=l_children_ids_lst,
            jac_bool=virtual_jac,
        )
        virtual_ll_dq = virtual_ll_grad_for_nontop_layers(
            nontop_ids_lst=nontop_ids_lst,
            ex_id=ex_id,
            this_example_samples=this_example_samples,
            this_example_virtual_samples=this_example_virtual_samples,
            children_ids_lst=l_children_ids_lst,
            jac_bool=virtual_jac,
        )

    if virtual_wrt_real_jac:
        (
            virtual_top_ll_wrt_real_dq,
            virtual_top_grad_mu_wrt_real_dq,
            virtual_top_grad_wrt_real_dq,
        ) = virtual_ll_grad_for_top_layers(
            top_ids_lst=top_ids_lst,
            ex_id=ex_id,
            this_example_samples=this_example_samples,
            this_example_virtual_samples=this_example_samples,
            children_ids_lst=l_children_ids_lst,
            jac_bool=virtual_wrt_real_jac,
        )

        (
            virtual_ll_wrt_real_dq,
            virtual_grad_wrt_real_dq,
            virtual_grad_mu_wrt_real_dq,
        ) = virtual_ll_grad_for_nontop_layers(
            nontop_ids_lst=nontop_ids_lst,
            ex_id=ex_id,
            this_example_samples=this_example_samples,
            this_example_virtual_samples=this_example_samples,
            children_ids_lst=l_children_ids_lst,
            jac_bool=virtual_wrt_real_jac,
        )

        for virtual_grad_mu_dq_sample in virtual_grad_mu_wrt_real_dq:
            virtual_grad_mu_dq_sample += deque([0] * len(evidences_ids_set))
        virtual_grad_wrt_real_arr = organize_virtual_grad(
            virtual_top_grad_dq=virtual_top_grad_wrt_real_dq,
            virtual_grad_dq=virtual_grad_wrt_real_dq,
            virtual_grad_mu_dq=virtual_grad_mu_wrt_real_dq,
        )
    else:
        virtual_top_ll_wrt_real_dq = virtual_ll_grad_for_top_layers(
            top_ids_lst=top_ids_lst,
            ex_id=ex_id,
            this_example_samples=this_example_samples,
            this_example_virtual_samples=this_example_samples,
            children_ids_lst=l_children_ids_lst,
            jac_bool=virtual_wrt_real_jac,
        )
        virtual_ll_wrt_real_dq = virtual_ll_grad_for_nontop_layers(
            nontop_ids_lst=nontop_ids_lst,
            ex_id=ex_id,
            this_example_samples=this_example_samples,
            this_example_virtual_samples=this_example_samples,
            children_ids_lst=l_children_ids_lst,
            jac_bool=virtual_wrt_real_jac,
        )

    if jac:
        ll_dq, grad_dq = ll_grad_for_nontop_layers(
            nontop_ids_lst=nontop_ids_lst,
            ex_id=ex_id,
            this_example_samples=layers_samples[ex_id],
            jac_bool=jac,
        )
        grad_dq = organize_grad(grad_dq)
    else:
        ll_dq = ll_grad_for_nontop_layers(
            nontop_ids_lst=nontop_ids_lst,
            ex_id=ex_id,
            this_example_samples=layers_samples[ex_id],
            jac_bool=jac,
        )

    stats_parallel = Stats_parallel_collections(
        top_ll_lst=top_ll_lst,
        top_mu_grad_lst=top_mu_grad_lst,
        top_base_rate_lst=top_base_rate_lst,
        ll_dq=ll_dq,
        grad_dq=grad_dq,
        virtual_top_ll_dq=virtual_top_ll_dq,
        virtual_top_ll_wrt_real_dq=virtual_top_ll_wrt_real_dq,
        virtual_top_grad_mu_dq=virtual_top_grad_mu_dq,
        virtual_top_grad_mu_wrt_real_dq=virtual_top_grad_mu_wrt_real_dq,
        virtual_ll_dq=virtual_ll_dq,
        virtual_ll_wrt_real_dq=virtual_ll_wrt_real_dq,
        virtual_grad_arr=virtual_grad_arr,
        virtual_grad_wrt_real_arr=virtual_grad_wrt_real_arr,
    )

    return stats_parallel


def organize_virtual_grad(virtual_top_grad_dq, virtual_grad_dq, virtual_grad_mu_dq):
    virtual_grad_all_dq = deque()
    virtual_grad_all_dq_append = virtual_grad_all_dq.append
    for virtual_top_grad_dict, virtual_grad_dict, virtual_grad_mu_dq_sample in zip(
        virtual_top_grad_dq, virtual_grad_dq, virtual_grad_mu_dq
    ):
        virtual_grad_sample_dict = {**virtual_top_grad_dict, **virtual_grad_dict}
        virtual_grad_sample_dq = deque()
        virtual_grad_sample_dq_append = virtual_grad_sample_dq.append
        for layer_id in nontop_ids_lst:
            if virtual_grad_mu_dq_sample:
                virtual_grad_sample_dq_append(virtual_grad_mu_dq_sample.popleft())
            parents_ids_dict = parents_ids_lst[layer_id]
            for p_id in parents_ids_dict:
                virtual_grad_sample_dq += virtual_grad_sample_dict[(p_id, layer_id)]
        virtual_grad_all_dq_append(virtual_grad_sample_dq)
    virtual_grad_arr = np_array(virtual_grad_all_dq)
    return virtual_grad_arr


def organize_grad(grad_dq):
    grad_all_dq = deque()
    grad_all_dq_append = grad_all_dq.append
    real_kernel_layer_ids = [
        layer_id for layer_id in nontop_ids_lst if layer_id not in evidences_ids_set
    ]
    real_kernel_layer_ids = top_ids_lst + real_kernel_layer_ids
    for grad_dq_dict in grad_dq:
        grad_sample_dq = deque()
        for layer_id in real_kernel_layer_ids:
            children_ids_dict = children_ids_lst[layer_id]
            for c_id in children_ids_dict:
                grad_sample_dq += grad_dq_dict[(c_id, layer_id)]
        grad_all_dq_append(grad_sample_dq)
    return grad_all_dq


def ll_grad_for_top_layers(top_ids_lst, ex_id, jac_bool):

    end_time = end_time_tuple[ex_id]

    num_of_events_each_example_top_layers = np_array(
        [
            [len(layers_samples[ex_id][layer_id][c]) for layer_id in top_ids_lst]
            for c in range(num_of_mcmc_samples)
        ]
    )
    if valid:
        base_rate_each_example_top_layers = np_array(
            [layers[layer_id].valid_base_rate[ex_id] for layer_id in top_ids_lst]
        )
    else:
        base_rate_each_example_top_layers = np_array(
            [layers[layer_id].base_rate[ex_id] for layer_id in top_ids_lst]
        )
    if np_any(base_rate_each_example_top_layers == 0):
        zero_base_rate_mask = base_rate_each_example_top_layers == 0
        zero_num_of_events = num_of_events_each_example_top_layers == 0

        assert not np_any(zero_base_rate_mask != zero_num_of_events)
        fake_base_rate_each_example_top_layers = copy.deepcopy(
            base_rate_each_example_top_layers
        )
        fake_base_rate_each_example_top_layers[zero_base_rate_mask] += 1e-6
        ll_events = num_of_events_each_example_top_layers * np_log(
            fake_base_rate_each_example_top_layers
        )
        if jac_bool:
            grad = (
                -end_time
                + num_of_events_each_example_top_layers
                / fake_base_rate_each_example_top_layers
            ) * sp.grad_inv_fun(base_rate_each_example_top_layers)

    else:
        ll_events = num_of_events_each_example_top_layers * np_log(
            base_rate_each_example_top_layers
        )

        if jac_bool:
            grad = (
                -end_time
                + num_of_events_each_example_top_layers
                / base_rate_each_example_top_layers
            ) * sp.grad_inv_fun(base_rate_each_example_top_layers)
    ll_time = -end_time * base_rate_each_example_top_layers
    ll = ll_events + ll_time

    if maximize:
        base_rate = num_of_events_each_example_top_layers / end_time

    if jac_bool and maximize:
        return ll, grad, base_rate
    elif jac_bool:
        return ll, grad
    elif maximize:
        return ll, base_rate
    else:
        return ll


def virtual_ll_grad_for_top_layers(
    top_ids_lst,
    ex_id,
    this_example_samples,
    this_example_virtual_samples,
    children_ids_lst,
    jac_bool,
):
    if valid:
        virtual_base_rate_each_example_top_layers = {
            layer_id: layers[layer_id].virtual_valid_base_rate[ex_id]
            for layer_id in top_ids_lst
        }
    else:
        virtual_base_rate_each_example_top_layers = {
            layer_id: layers[layer_id].virtual_base_rate[ex_id]
            for layer_id in top_ids_lst
        }
    grad_mu_dq = deque()
    ll_dq = deque()
    grad_dq = deque()
    grad_mu_dq_append = grad_mu_dq.append
    ll_dq_append = ll_dq.append
    grad_dq_append = grad_dq.append
    end_time = end_time_tuple[ex_id]
    for count in range(num_of_mcmc_samples):
        ll_dq_sample = deque()
        grad_dict_sample = {}
        grad_mu_dq_sample = deque()
        ll_dq_sample_append = ll_dq_sample.append
        grad_mu_dq_sample_append = grad_mu_dq_sample.append
        for layer_id in top_ids_lst:
            children_ids_dict = children_ids_lst[layer_id]

            lower_layers_lst = [layers[c_id] for c_id in children_ids_dict]

            # this layer pos in lower layer lst
            stats_lst_ids_lst = [
                parents_ids_lst[c_id][layer_id] for c_id in children_ids_dict
            ]

            this_layer_virtual_samples = this_example_virtual_samples[layer_id]
            stats = collect_stats_virtual_layers(
                layer_id=layer_id,
                this_layer_virtual_samples=this_layer_virtual_samples,
                this_example_samples=this_example_samples,
                lower_layers_lst=lower_layers_lst,
                count=count,
                ex_id=ex_id,
                stats_lst_ids_lst=stats_lst_ids_lst,
                jac_bool=jac_bool,
            )

            virtual_base_rate = virtual_base_rate_each_example_top_layers[layer_id]
            ll_time = (
                -np_sum(
                    [
                        np_sum(expec)
                        for expec in stats.children_events_time_to_beginning_expec_with_syn_end_lst
                    ]
                )
                - virtual_base_rate * end_time
            )
            ll_events = np_sum(np_log(stats.virtual_intensity_array))
            ll_layer = ll_time + ll_events
            if jac_bool:
                grad_dict = kernel_grad(
                    layer_id=layer_id,
                    var_ids=layers[layer_id].virtual_var_ids,
                    stats=stats,
                    parents_ids_dict=children_ids_dict,
                    virtual=True,
                    stats_lst_ids_lst=stats_lst_ids_lst,
                    kernel_type=lower_layers_lst[0].virtual_kernels_type_lst[0],
                )
                grad_dict_sample = {**grad_dict_sample, **grad_dict}
                grad_mu_dq_sample_append(
                    (-end_time + np_sum(1 / stats.virtual_intensity_array))
                    * sp.grad_inv_fun(virtual_base_rate)
                )
            ll_dq_sample_append(ll_layer)
        ll_dq_append(ll_dq_sample)
        if jac_bool:
            grad_mu_dq_append(grad_mu_dq_sample)
            grad_dq_append(grad_dict_sample)
    if jac_bool:
        return ll_dq, grad_mu_dq, grad_dq
    else:
        return ll_dq


def virtual_ll_grad_for_nontop_layers(
    nontop_ids_lst,
    ex_id,
    this_example_samples,
    this_example_virtual_samples,
    children_ids_lst,
    jac_bool,
):
    ll_dq = deque()
    grad_dq = deque()
    virtual_grad_mu_dq = deque()
    ll_dq_append = ll_dq.append
    grad_dq_append = grad_dq.append
    virtual_grad_mu_dq_append = virtual_grad_mu_dq.append

    end_time = end_time_tuple[ex_id]
    for count in range(num_of_mcmc_samples):
        ll_dq_sample = deque()
        grad_dict_sample = {}
        virtual_grad_mu_dq_sample = deque()
        ll_dq_sample_append = ll_dq_sample.append
        virtual_grad_mu_dq_sample_append = virtual_grad_mu_dq_sample.append
        for layer_id in nontop_ids_lst:
            if layer_id in evidences_ids_set:
                break
            children_ids_dict = children_ids_lst[layer_id]
            lower_layers_lst = [layers[c_id] for c_id in children_ids_dict]

            stats_lst_ids_lst = [
                parents_ids_lst[c_id][layer_id] for c_id in children_ids_dict
            ]

            this_layer_virtual_samples = this_example_virtual_samples[layer_id]
            stats = collect_stats_virtual_layers(
                layer_id=layer_id,
                this_layer_virtual_samples=this_layer_virtual_samples,
                this_example_samples=this_example_samples,
                lower_layers_lst=lower_layers_lst,
                count=count,
                ex_id=ex_id,
                stats_lst_ids_lst=stats_lst_ids_lst,
                jac_bool=jac_bool,
            )

            virtual_base_rate = layers[layer_id].virtual_base_rate
            ll_time = (
                -np_sum(
                    [
                        np_sum(expec)
                        for expec in stats.children_events_time_to_beginning_expec_with_syn_end_lst
                    ]
                )
                - virtual_base_rate * end_time
            )
            ll_events = np_sum(np_log(stats.virtual_intensity_array))
            ll_layer = ll_time + ll_events

            if jac_bool:
                var_ids = layers[layer_id].virtual_var_ids
                if var_ids[0] == 0:
                    virtual_grad_mu_dq_sample_append(
                        (-end_time + np_sum(1 / stats.virtual_intensity_array))
                        * sp.grad_inv_fun(virtual_base_rate)
                    )
                grad_dict = kernel_grad(
                    layer_id=layer_id,
                    var_ids=var_ids,
                    stats=stats,
                    parents_ids_dict=children_ids_dict,
                    virtual=True,
                    stats_lst_ids_lst=stats_lst_ids_lst,
                    kernel_type=lower_layers_lst[0].virtual_kernels_type_lst[0],
                )
                grad_dict_sample = {**grad_dict_sample, **grad_dict}
            ll_dq_sample_append(ll_layer)
        ll_dq_append(ll_dq_sample)
        if jac_bool:
            grad_dq_append(grad_dict_sample)
            virtual_grad_mu_dq_append(virtual_grad_mu_dq_sample)
    if jac_bool:
        return ll_dq, grad_dq, virtual_grad_mu_dq
    else:
        return ll_dq


def ll_grad_for_nontop_layers(nontop_ids_lst, ex_id, this_example_samples, jac_bool):
    ll_dq = deque()
    grad_dq = deque()
    ll_dq_append = ll_dq.append
    grad_dq_append = grad_dq.append

    for count in range(num_of_mcmc_samples):
        ll_dq_sample = deque()
        grad_dict_sample = {}
        ll_dq_sample_append = ll_dq_sample.append
        for layer_id in nontop_ids_lst:
            parents_ids_dict = parents_ids_lst[layer_id]
            upper_layers_lst = [layers[p_id] for p_id in parents_ids_dict]
            stats_lst_ids_lst = [
                children_ids_lst[u_id][layer_id] for u_id in parents_ids_dict
            ]
            # try:
            #     this_layer_samples = this_example_samples[layer_id]
            # except KeyError:
            #     this_layer_samples = evidences_dict[ex_id]

            # easier to debug
            if layer_id in this_example_samples:
                this_layer_samples = this_example_samples[layer_id]
            else:
                this_layer_samples = evidences_dict[ex_id][layer_id]

            stats = collect_stats_nontop_layers(
                layer_id=layer_id,
                this_layer_samples=this_layer_samples,
                this_example_samples=this_example_samples,
                upper_layers_lst=upper_layers_lst,
                count=count,
                ex_id=ex_id,
                stats_lst_ids_lst=stats_lst_ids_lst,
                jac_bool=jac_bool,
            )

            ll_time = -np_sum(
                [np_sum(expec) for expec in stats.parents_events_time_to_end_expec_lst]
            )
            ll_events = np_sum(np_log(stats.intensity_array))
            ll_layer = ll_time + ll_events
            if jac_bool:
                grad_dict = kernel_grad(
                    layer_id=layer_id,
                    var_ids=layers[layer_id].var_ids,
                    stats=stats,
                    parents_ids_dict=parents_ids_dict,
                    virtual=False,
                    stats_lst_ids_lst=stats_lst_ids_lst,
                    kernel_type=upper_layers_lst[0].kernels_type_lst[0],
                )
                grad_dict_sample = {**grad_dict_sample, **grad_dict}
            ll_dq_sample_append(ll_layer)
        ll_dq_append(ll_dq_sample)
        if jac_bool:
            grad_dq_append(grad_dict_sample)
    if jac_bool:
        return ll_dq, grad_dq
    else:
        return ll_dq


def collect_stats_nontop_layers(
    layer_id,
    this_layer_samples,
    this_example_samples,
    upper_layers_lst,
    count,
    ex_id,
    stats_lst_ids_lst,
    jac_bool,
):
    if isinstance(this_layer_samples, list):
        this_layer_sample = this_layer_samples[count]
    else:
        this_layer_sample = this_layer_samples
    parents_events_time_to_end_lst = [
        end_time_tuple[ex_id] - this_example_samples[p_layer.layer_id][count]
        for p_layer in upper_layers_lst
    ]
    this_layer = layers[layer_id]

    parents_events_time_to_end_expec_lst = [
        upper_layer.kernels_lst[stats_lst_id].expectation(
            input_arr=parents_events_time_to_end, start_time=0
        )
        for upper_layer, parents_events_time_to_end, stats_lst_id in zip(
            upper_layers_lst, parents_events_time_to_end_lst, stats_lst_ids_lst
        )
    ]
    p_min_lst = [
        np_min(this_example_samples[p_layer.layer_id][count])
        if len(this_example_samples[p_layer.layer_id][count]) != 0
        else np_inf
        for p_layer in upper_layers_lst
    ]
    p_min = np_min(p_min_lst)
    intensity_array = 0
    events_intervals_lst = []
    valid_ids_lst = []
    u_layer_kernel_type = upper_layers_lst[0].kernels_type_lst[0]
    if u_layer_kernel_type == "ExpKernel2":
        kernel_rate_pos = 1
        exp_intervals_div_intensity_lst = []
    elif u_layer_kernel_type in ("GammaKernel", "GammaKernelWithShift"):
        kernel_rate_pos = 2
        exp_intervals_div_intensity_lst = []
    elif u_layer_kernel_type == "PowerLawKernel":
        power_rate_pos = 1
        power_shift_pos = 2
        power_intervals_div_intensity_lst = []
    for stats_count, p_layer in enumerate(upper_layers_lst):
        stats_lst_id = stats_lst_ids_lst[stats_count]
        p_id = p_layer.layer_id
        this_layer_sub_upper_layer = (
            this_layer_sample[:, None] - this_example_samples[p_id][count]
        )
        empty_ids = this_layer_sub_upper_layer <= 0
        valid_ids = np_logical_not(empty_ids)
        valid_ids_lst.append(valid_ids)
        this_layer_sub_upper_layer[empty_ids] = np.inf
        intensity_mat = p_layer.kernels_lst[stats_lst_id].fun(
            this_layer_sub_upper_layer, 0
        )
        intensity_array += np_sum(intensity_mat, axis=1)
        if jac_bool:
            events_intervals_lst.append(this_layer_sub_upper_layer[valid_ids])
            if u_layer_kernel_type in (
                "ExpKernel2",
                "GammaKernel",
                "GammaKernelWithShift",
            ):
                exp_intervals_div_intensity_lst.append(
                    np_exp(
                        -p_layer.kernels_param_lst[stats_lst_id][kernel_rate_pos]
                        * this_layer_sub_upper_layer
                    )
                )
            elif u_layer_kernel_type == "PowerLawKernel":
                power_intervals_div_intensity_lst.append(
                    (
                        this_layer_sub_upper_layer
                        + p_layer.kernels_param_lst[stats_lst_id][power_shift_pos]
                    )
                    ** (-1 - p_layer.kernels_param_lst[stats_lst_id][power_rate_pos])
                )
    intensity_array += this_layer.base_rate
    nonzero_entry = this_layer_sample > p_min
    intensity_array[np_logical_and(nonzero_entry, intensity_array < 1e-100)] = 1e-100
    if jac_bool:
        if u_layer_kernel_type in ("ExpKernel2", "GammaKernel", "GammaKernelWithShift"):
            exp_intervals_div_intensity_lst = [
                (value / intensity_array[:, None])[valid_ids]
                for valid_ids, value in zip(
                    valid_ids_lst, exp_intervals_div_intensity_lst
                )
            ]

            grad_stats = Expkernel2gradstats(
                parents_events_time_to_end_expec_lst=parents_events_time_to_end_expec_lst,
                intensity_array=intensity_array,
                parents_events_time_to_end_lst=parents_events_time_to_end_lst,
                events_intervals_lst=events_intervals_lst,
                exp_intervals_div_intensity_lst=exp_intervals_div_intensity_lst,
            )
        elif u_layer_kernel_type == "PowerLawKernel":
            power_intervals_div_intensity_lst = [
                (value / intensity_array[:, None])[valid_ids]
                for valid_ids, value in zip(
                    valid_ids_lst, power_intervals_div_intensity_lst
                )
            ]

            grad_stats = Powerlawkernelgradstats(
                parents_events_time_to_end_expec_lst=parents_events_time_to_end_expec_lst,
                intensity_array=intensity_array,
                parents_events_time_to_end_lst=parents_events_time_to_end_lst,
                events_intervals_lst=events_intervals_lst,
                power_intervals_div_intensity_lst=power_intervals_div_intensity_lst,
            )
        return grad_stats
    else:
        ll_stats = Llstats(
            parents_events_time_to_end_expec_lst=parents_events_time_to_end_expec_lst,
            intensity_array=intensity_array,
        )
        return ll_stats


def collect_stats_virtual_layers(
    layer_id,
    this_layer_virtual_samples,
    this_example_samples,
    lower_layers_lst,
    count,
    ex_id,
    stats_lst_ids_lst,
    jac_bool,
):
    end_time = end_time_tuple[ex_id]
    this_layer_virtual_sample = this_layer_virtual_samples[count]
    children_events_time_to_beginning_lst = [
        this_example_samples[c_layer.layer_id][count]
        if c_layer.layer_id in this_example_samples
        else evidences_dict[ex_id][c_layer.layer_id]
        for c_layer in lower_layers_lst
    ]

    children_events_time_to_beginning_with_syn_end_lst = [
        np_append(s, end_time) if c_layer.synthetic_event_end else s
        for s, c_layer in zip(children_events_time_to_beginning_lst, lower_layers_lst)
    ]
    children_events_time_to_beginning_expec_with_syn_end_lst = [
        lower_layer.virtual_kernels_lst[stats_lst_id].expectation(
            input_arr=children_events_time_to_beginning_with_syn_end, start_time=0
        )
        for lower_layer, children_events_time_to_beginning_with_syn_end, stats_lst_id in zip(
            lower_layers_lst,
            children_events_time_to_beginning_with_syn_end_lst,
            stats_lst_ids_lst,
        )
    ]
    virtual_intensity_array = 0
    events_intervals_lst = []
    valid_ids_lst = []

    c_layer_kernel_type = lower_layers_lst[0].virtual_kernels_type_lst[0]
    if c_layer_kernel_type == "ExpKernel2":
        kernel_rate_pos = 1
        exp_intervals_div_intensity_lst = []
    elif c_layer_kernel_type in ("GammaKernel", "GammaKernelWithShift"):
        kernel_rate_pos = 2
        exp_intervals_div_intensity_lst = []
    elif c_layer_kernel_type == "PowerLawKernel":
        power_rate_pos = 1
        power_shift_pos = 2
        power_intervals_div_intensity_lst = []
    for lower_layer_count, c_layer in enumerate(lower_layers_lst):
        lower_layer_sample = children_events_time_to_beginning_with_syn_end_lst[
            lower_layer_count
        ]
        stats_lst_id = stats_lst_ids_lst[lower_layer_count]

        lower_layer_sub_this_virtual_layer = (
            lower_layer_sample[:, None] - this_layer_virtual_sample
        )
        empty_ids = lower_layer_sub_this_virtual_layer <= 0
        valid_ids = np_logical_not(empty_ids)
        valid_ids_lst.append(valid_ids)
        lower_layer_sub_this_virtual_layer[empty_ids] = np.inf

        intensity_mat = c_layer.virtual_kernels_lst[stats_lst_id].fun(
            lower_layer_sub_this_virtual_layer, 0
        )
        virtual_intensity_array += np_sum(intensity_mat, axis=0)
        if jac_bool:
            events_intervals_lst.append(lower_layer_sub_this_virtual_layer[valid_ids])
            if c_layer_kernel_type in (
                "ExpKernel2",
                "GammaKernel",
                "GammaKernelWithShift",
            ):
                exp_intervals_div_intensity_lst.append(
                    np_exp(
                        -c_layer.virtual_kernels_param_lst[stats_lst_id][
                            kernel_rate_pos
                        ]
                        * lower_layer_sub_this_virtual_layer
                    )
                )
            elif c_layer_kernel_type == "PowerLawKernel":
                power_intervals_div_intensity_lst.append(
                    (
                        lower_layer_sub_this_virtual_layer
                        + c_layer.virtual_kernels_param_lst[stats_lst_id][
                            power_shift_pos
                        ]
                    )
                    ** (
                        -1
                        - c_layer.virtual_kernels_param_lst[stats_lst_id][
                            power_rate_pos
                        ]
                    )
                )
    if hasattr(layers[layer_id].virtual_base_rate, "__len__"):
        if valid:
            virtual_intensity_array += layers[layer_id].virtual_valid_base_rate[ex_id]
        else:
            virtual_intensity_array += layers[layer_id].virtual_base_rate[ex_id]
    else:
        virtual_intensity_array += layers[layer_id].virtual_base_rate
    virtual_intensity_array[virtual_intensity_array < 1e-100] = 1e-100
    if jac_bool:
        if c_layer_kernel_type in ("ExpKernel2", "GammaKernel", "GammaKernelWithShift"):
            exp_intervals_div_intensity_lst = [
                (value / virtual_intensity_array[None, :])[valid_ids]
                for valid_ids, value in zip(
                    valid_ids_lst, exp_intervals_div_intensity_lst
                )
            ]

            grad_stats = Expkernel2grad_virtual_stats(
                children_events_time_to_beginning_expec_with_syn_end_lst=children_events_time_to_beginning_expec_with_syn_end_lst,
                virtual_intensity_array=virtual_intensity_array,
                children_events_time_to_beginning_with_syn_end_lst=children_events_time_to_beginning_with_syn_end_lst,
                events_intervals_lst=events_intervals_lst,
                exp_intervals_div_intensity_lst=exp_intervals_div_intensity_lst,
            )
        elif c_layer_kernel_type == "PowerLawKernel":
            power_intervals_div_intensity_lst = [
                (value / virtual_intensity_array[None, :])[valid_ids]
                for valid_ids, value in zip(
                    valid_ids_lst, power_intervals_div_intensity_lst
                )
            ]

            grad_stats = Powerlawkernelgrad_virtual_stats(
                children_events_time_to_beginning_expec_with_syn_end_lst=children_events_time_to_beginning_expec_with_syn_end_lst,
                virtual_intensity_array=virtual_intensity_array,
                children_events_time_to_beginning_with_syn_end_lst=children_events_time_to_beginning_with_syn_end_lst,
                events_intervals_lst=events_intervals_lst,
                power_intervals_div_intensity_lst=power_intervals_div_intensity_lst,
            )
        return grad_stats
    else:
        ll_stats = Ll_virtual_stats(
            children_events_time_to_beginning_expec_with_syn_end_lst=children_events_time_to_beginning_expec_with_syn_end_lst,
            virtual_intensity_array=virtual_intensity_array,
        )
        return ll_stats


def parallel_ll_grad(example_ids, parallel=True):
    if parallel:
        pool = Pool()
        stats_parallel_lst = pool.map(ll_grad_for_all_layers, example_ids)
        pool.close()
        pool.join()
        pool = None
    else:
        stats_parallel_lst = list(map(ll_grad_for_all_layers, example_ids))
    if jac:
        reduced_var_grad = None
        reduced_var_virtual_grad = None
        return stats_parallel_lst, reduced_var_grad, reduced_var_virtual_grad
    else:
        return stats_parallel_lst


def kernel_grad(
    layer_id, var_ids, stats, parents_ids_dict, virtual, stats_lst_ids_lst, kernel_type,
):
    grad_layer_dict = {}
    if kernel_type in ("ExpKernel2", "GammaKernel", "GammaKernelWithShift"):
        exp_intervals_div_intensity_lst = stats.exp_intervals_div_intensity_lst
    elif kernel_type == "PowerLawKernel":
        power_intervals_div_intensity_lst = stats.power_intervals_div_intensity_lst
    events_intervals_lst = stats.events_intervals_lst
    for p_id, stats_lst_id, kernel_stats_lst_id in zip(
        parents_ids_dict.keys(), parents_ids_dict.values(), stats_lst_ids_lst
    ):
        if not virtual:
            p_layer_kernel_param = layers[p_id].kernels_param_lst[kernel_stats_lst_id]
            parents_events_time_to_end_lst = stats.parents_events_time_to_end_lst
        else:
            p_layer_kernel_param = layers[p_id].virtual_kernels_param_lst[
                kernel_stats_lst_id
            ]
            parents_events_time_to_end_lst = (
                stats.children_events_time_to_beginning_with_syn_end_lst
            )
        grad_layer_dq = deque()
        if kernel_type == "ExpKernel2":
            alpha = p_layer_kernel_param[0]
            beta = p_layer_kernel_param[1]
            events_intervals = events_intervals_lst[stats_lst_id]
            parents_events_time_to_end = parents_events_time_to_end_lst[stats_lst_id]
            exp_intervals_div_intensity = exp_intervals_div_intensity_lst[stats_lst_id]
            exp_end_time = np_exp(-beta * parents_events_time_to_end)
            for g_id in var_ids:
                if g_id == 1:
                    alpha_grad_first_part = -np_sum(1 - exp_end_time)
                    alpha_grad_second_part = np_sum(beta * exp_intervals_div_intensity)
                    alpha_grad = alpha_grad_first_part + alpha_grad_second_part
                    alpha_grad *= sp.grad_inv_fun(alpha)
                    grad_layer_dq.append(alpha_grad)
                elif g_id == 2:
                    beta_grad_first_part = -np_sum(
                        parents_events_time_to_end * exp_end_time
                    )
                    beta_grad_second_part = np_sum(
                        exp_intervals_div_intensity * (1 - beta * events_intervals)
                    )
                    beta_grad = alpha * (beta_grad_first_part + beta_grad_second_part)
                    beta_grad *= sp.grad_inv_fun(beta)
                    grad_layer_dq.append(beta_grad)
            grad_layer_dict[(layer_id, p_id)] = grad_layer_dq

        if kernel_type == "GammaKernel":
            events_intervals = events_intervals_lst[stats_lst_id]
            exp_intervals_div_intensity = exp_intervals_div_intensity_lst[stats_lst_id]
            parents_events_time_to_end = parents_events_time_to_end_lst[stats_lst_id]
            partition = p_layer_kernel_param[0]
            shape = p_layer_kernel_param[1]
            rate = p_layer_kernel_param[2]
            rate_events_intervals = rate * events_intervals
            rate_end_time = rate * parents_events_time_to_end
            exp_end_time = np_exp(-rate_end_time)
            gamma_shape = gamma(shape)
            gammainc_shape_rate_end_time = gammainc(shape, rate_end_time)
            rate_events_intervals_pow_shape_m1 = rate_events_intervals ** (shape - 1)

            for g_id in var_ids:
                if g_id == 1:
                    partition_grad_first_part = -np_sum(gammainc_shape_rate_end_time)
                    partition_grad_second_part = np_sum(
                        rate ** shape
                        / gamma_shape
                        * events_intervals ** (shape - 1)
                        * exp_intervals_div_intensity
                    )
                    partition_grad = (
                        partition_grad_first_part + partition_grad_second_part
                    ) * sp.grad_inv_fun(partition)
                    grad_layer_dq.append(partition_grad)
                if g_id == 2:
                    # meijerg_A1 = []
                    # meijerg_A2 = [1] * 2
                    # meijerg_A = [meijerg_A1, meijerg_A2]
                    # meijerg_B1 = [0, 0, shape]
                    # meijerg_B2 = []
                    # meijerg_B = [meijerg_B1, meijerg_B2]

                    # def meijerg_grad(t):
                    #     return meijerg(meijerg_A, meijerg_B, t)

                    # meijerg_grad_arr = np_frompyfunc(meijerg_grad, 1, 1)
                    digamma_shape = digamma(shape)
                    # alpha_grad_first_part = np_sum(
                    #     (
                    #         digamma_shape * (gammainc_shape_rate_end_time - 1)
                    #         + np_log(rate_end_time) * gammaincc(shape, rate_end_time)
                    #         + rate_end_time
                    #         * meijerg_grad_arr(rate_end_time).astype(np_float64)
                    #         / gamma_shape
                    #     )
                    # )
                    alpha_grad_first_part = -np_sum(
                        (
                            gammainc(shape + 1e-9, rate_end_time)
                            - gammainc_shape_rate_end_time
                        )
                        / 1e-9
                    )
                    alpha_grad_second_part = (
                        np_sum(
                            rate_events_intervals_pow_shape_m1
                            * (np_log(rate_events_intervals) - digamma_shape)
                            * exp_intervals_div_intensity
                        )
                        / gamma_shape
                        * rate
                    )

                    alpha_grad = partition * (
                        alpha_grad_first_part + alpha_grad_second_part
                    )
                    alpha_grad *= sp.grad_inv_fun(shape)
                    grad_layer_dq.append(alpha_grad)
                elif g_id == 3:
                    fake_rate_end_time = copy.deepcopy(rate_end_time)
                    fake_rate_end_time[fake_rate_end_time == 0] = 1e-6

                    beta_grad_first_part = -np_sum(
                        (fake_rate_end_time) ** (shape - 1)
                        * exp_end_time
                        * parents_events_time_to_end
                    )
                    beta_grad_second_part = np_sum(
                        exp_intervals_div_intensity
                        * (
                            rate_events_intervals_pow_shape_m1
                            * (shape - rate * events_intervals)
                        )
                    )
                    beta_grad = (
                        partition
                        / gamma_shape
                        * (beta_grad_first_part + beta_grad_second_part)
                    )

                    beta_grad *= sp.grad_inv_fun(rate)
                    grad_layer_dq.append(beta_grad)
            grad_layer_dict[(layer_id, p_id)] = grad_layer_dq

        if kernel_type == "PowerLawKernel":
            alpha = p_layer_kernel_param[0]
            beta = p_layer_kernel_param[1]
            c = p_layer_kernel_param[2]
            events_intervals = events_intervals_lst[stats_lst_id]
            power_intervals_div_intensity = power_intervals_div_intensity_lst[
                stats_lst_id
            ]
            parents_events_time_to_end = parents_events_time_to_end_lst[stats_lst_id]
            end_time_shift = parents_events_time_to_end + c
            power_end_time_shift = end_time_shift ** (-1 - beta)
            events_intervals = events_intervals_lst[stats_lst_id]

            for g_id in var_ids:
                if g_id == 1:
                    alpha_grad_first_part = -np_sum(
                        c ** beta
                        * (-power_end_time_shift * end_time_shift + c ** (-beta))
                    )
                    alpha_grad_second_part = np_sum(
                        beta * c ** beta * power_intervals_div_intensity
                    )
                    alpha_grad = alpha_grad_first_part + alpha_grad_second_part
                    alpha_grad *= sp.grad_inv_fun(alpha)
                    grad_layer_dq.append(alpha_grad)
                elif g_id == 2:
                    beta_grad_first_part = alpha * np_log(
                        c
                    ) * alpha_grad_first_part - alpha * c ** beta * np_sum(
                        power_end_time_shift * end_time_shift * np_log(end_time_shift)
                        - c ** (-beta) * np_log(c)
                    )
                    beta_grad_second_part = (
                        alpha
                        * beta
                        * c ** beta
                        * np_sum(
                            ((np_log(c) + 1 / beta) - np_log(events_intervals + c))
                            * power_intervals_div_intensity
                        )
                    )
                    beta_grad = beta_grad_first_part + beta_grad_second_part
                    beta_grad *= sp.grad_inv_fun(beta)
                    grad_layer_dq.append(beta_grad)
                elif g_id == 3:
                    c_grad_first_part = (
                        alpha * beta / c * alpha_grad_first_part
                        - alpha
                        * c ** beta
                        * np_sum(beta * power_end_time_shift - beta * c ** (-1 - beta))
                    )
                    c_grad_second_part = np_sum(
                        alpha
                        * beta
                        * c ** (beta - 1)
                        * (beta - c * (1 + beta) / (events_intervals + c))
                        * power_intervals_div_intensity
                    )
                    c_grad = c_grad_first_part + c_grad_second_part
                    c_grad *= sp.grad_inv_fun(c)
                    grad_layer_dq.append(c_grad)
            grad_layer_dict[(layer_id, p_id)] = grad_layer_dq

        if kernel_type == "GammaKernelWithShift":
            events_intervals = events_intervals_lst[stats_lst_id]
            exp_intervals_div_intensity = exp_intervals_div_intensity_lst[stats_lst_id]
            parents_events_time_to_end = parents_events_time_to_end_lst[stats_lst_id]
            partition = p_layer_kernel_param[0]
            shape = p_layer_kernel_param[1]
            rate = p_layer_kernel_param[2]
            c = p_layer_kernel_param[3]
            gammainc_const = gammainc(shape, rate * c)
            gammaincc_const = gammaincc(shape, rate * c)
            gamma_shape = gamma(shape)
            gammaincc_const_unreg = gammaincc_const * gamma_shape
            rate_c = rate * c
            exp_rate_c = np_exp(-rate_c)
            rate_c_pow_shape_m_1 = rate_c ** (shape - 1)
            exp_rate_parents_events_time_to_end = np_exp(
                -rate * parents_events_time_to_end
            )
            exp_rate_c_parents_events_time_to_end = (
                exp_rate_c * exp_rate_parents_events_time_to_end
            )
            parents_events_time_to_end_with_shift = parents_events_time_to_end + c
            rate_times_parents_events_time_to_end_with_shift = rate * (
                parents_events_time_to_end_with_shift
            )
            events_intervals_with_shift = events_intervals + c
            rate_pow_shape = rate ** shape
            shape_m_1 = shape - 1
            events_intervals_with_shift_pow_shape_m_1 = (
                events_intervals_with_shift ** shape_m_1
            )

            for g_id in var_ids:
                if g_id == 1:
                    partition_grad_first_part = (
                        -np_sum(
                            (
                                gammainc(
                                    shape,
                                    rate_times_parents_events_time_to_end_with_shift,
                                )
                                - gammainc_const
                            )
                        )
                        / gammaincc_const
                    )
                    partition_grad_second_part = np_sum(
                        exp_rate_c
                        * rate_pow_shape
                        / gammaincc_const_unreg
                        * events_intervals_with_shift_pow_shape_m_1
                        * exp_intervals_div_intensity
                    )
                    partition_grad = (
                        partition_grad_first_part + partition_grad_second_part
                    )
                    partition_grad *= sp.grad_inv_fun(partition)
                    grad_layer_dq.append(partition_grad)

                if g_id == 2:
                    shape_grad_first_part = (
                        -partition
                        * (
                            np_sum(
                                (
                                    gammainc(
                                        shape + 1e-9,
                                        rate_times_parents_events_time_to_end_with_shift,
                                    )
                                    - gammainc(shape + 1e-9, rate * c)
                                )
                            )
                            / gammaincc(shape + 1e-9, rate * c)
                            + partition_grad_first_part
                        )
                        / 1e-9
                    )
                    gammaincc_grad = (
                        (gammaincc(shape + 1e-9, rate_c) - gammaincc_const) / 1e-9
                        + gammaincc_const * digamma(shape)
                    ) * gamma_shape
                    shape_grad_second_part = (
                        partition
                        * exp_rate_c
                        * rate_pow_shape
                        * np_sum(
                            events_intervals_with_shift_pow_shape_m_1
                            * (
                                np_log(rate * events_intervals_with_shift)
                                * gammaincc_const_unreg
                                - gammaincc_grad
                            )
                            * exp_intervals_div_intensity
                        ) / gammaincc_const_unreg ** 2
                    )
                    shape_grad = shape_grad_first_part + shape_grad_second_part
                    shape_grad *= sp.grad_inv_fun(shape)
                    grad_layer_dq.append(shape_grad)

                if g_id == 3:
                    rate_grad_first_part = (
                        partition
                        * (
                            rate_c_pow_shape_m_1
                            * exp_rate_c
                            * c
                            * partition_grad_first_part
                            - np_sum(
                                rate_times_parents_events_time_to_end_with_shift
                                ** shape_m_1
                                * exp_rate_c_parents_events_time_to_end
                                * parents_events_time_to_end_with_shift
                                - rate_c_pow_shape_m_1 * exp_rate_c * c
                            )
                        )
                        / gammaincc_const_unreg
                    )
                    rate_grad_second_part = (
                        partition
                        * exp_rate_c
                        * rate_pow_shape
                        * np_sum(
                            events_intervals_with_shift_pow_shape_m_1
                            * (
                                rate_c_pow_shape_m_1
                                * exp_rate_c
                                * c
                                / gammaincc_const_unreg
                                + shape / rate
                                - c
                                - events_intervals
                            )
                            * exp_intervals_div_intensity
                        )
                        / gammaincc_const_unreg
                    )
                    rate_grad = rate_grad_first_part + rate_grad_second_part
                    rate_grad *= sp.grad_inv_fun(rate)
                    grad_layer_dq.append(rate_grad)

                if g_id == 4:
                    c_grad_first_part = (
                        partition
                        * (
                            rate_c_pow_shape_m_1
                            * exp_rate_c
                            * rate
                            * partition_grad_first_part
                            - np_sum(
                                rate_times_parents_events_time_to_end_with_shift
                                ** shape_m_1
                                * exp_rate_c_parents_events_time_to_end
                                * rate
                                - rate_c_pow_shape_m_1 * exp_rate_c * rate
                            )
                        )
                        / gammaincc_const_unreg
                    )
                    c_grad_second_part = (
                        partition
                        * rate_pow_shape
                        * np_sum(
                            (
                                rate_c_pow_shape_m_1
                                * exp_rate_c
                                * rate
                                / gammaincc_const_unreg
                                + shape_m_1 / events_intervals_with_shift
                                - rate
                            )
                            * exp_rate_c
                            * events_intervals_with_shift_pow_shape_m_1
                            * exp_intervals_div_intensity
                        )
                        / gammaincc_const_unreg
                    )
                    c_grad = c_grad_first_part + c_grad_second_part
                    c_grad *= sp.grad_inv_fun(c)
                    grad_layer_dq.append(c_grad)
            grad_layer_dict[(layer_id, p_id)] = grad_layer_dq
    return grad_layer_dict
