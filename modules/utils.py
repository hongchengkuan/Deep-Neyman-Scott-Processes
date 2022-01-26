from DPPLayers import (
    calc_virtual_intensity_mat_lst,
    np_max,
    calc_intensity_mat_lst,
)
import numpy as np
from collections import namedtuple, deque
from numpy.random import default_rng

np_concatenate = np.concatenate
np_append = np.append
np_min = np.min
np_log = np.log
np_sum = np.sum
np_array = np.array
np_where = np.where
np_logical_and = np.logical_and
np_arange = np.arange
np_any = np.any
np_inf = np.inf
np_logical_not = np.logical_not
np_ones = np.ones
quasi_layer = namedtuple(
    "quasi_layer", ["real_events", "kernels_lst", "virtual_kernels_lst"]
)

class BatchManager:
    def __init__(self, batch_size, num_of_evidences, num_of_epochs):
        self.batch_size = batch_size if batch_size > 0 else num_of_evidences
        self.num_of_iters = num_of_evidences // self.batch_size
        self.example_ids = list(range(num_of_evidences))
        self.num_of_epochs = num_of_epochs
        self.rng = default_rng(12345)

    def shuffle(self):
        self.rng.shuffle(self.example_ids)

    def example_ids_for_iter(self, i):
        return self.example_ids[i * self.batch_size : (i + 1) * self.batch_size]

def del_event_lower_layers_stats_diff(
    this_layer, lower_layers_lst, chosen_index, parents_ids_lst
):
    this_layer_real_events = this_layer.real_events
    to_del_event = this_layer_real_events[chosen_index]
    mask = np_ones(len(this_layer_real_events), dtype=bool)
    this_layer_id = this_layer.layer_id
    mask[chosen_index] = False
    remaining_events = this_layer_real_events[mask]
    stats_diff_dq = deque()
    l_np_min = np_min
    l_np_any = np_any
    l_np_logical_not = np_logical_not
    l_np_logical_and = np_logical_and
    l_np_sum = np_sum
    l_np_log = np_log
    stats_diff_dq_append = stats_diff_dq.append
    ll_diff_dq = deque()
    ll_diff_dq_append = ll_diff_dq.append
    if len(remaining_events) == 0:
        p_min_after = np_inf
    else:
        p_min_after = l_np_min(remaining_events)
    for lower_layer in lower_layers_lst:
        influenced_id_mask = lower_layer.real_events > to_del_event
        intensity_arr_before_del = lower_layer.intensity_arr[influenced_id_mask]

        # this layer position in lower layer lst
        lower_layer_parents_ids_dict = parents_ids_lst[lower_layer.layer_id]
        stats_lst_id = lower_layer_parents_ids_dict[this_layer_id]

        intensity_arr_after_del = (
            intensity_arr_before_del
            - lower_layer.intensity_mat_lst[stats_lst_id][
                influenced_id_mask, chosen_index
            ]
        )
        nonzero_entry_after = lower_layer.real_events[influenced_id_mask] > p_min_after

        if l_np_any(l_np_logical_not(nonzero_entry_after)):
            ll_events_diff = -np_inf
        else:
            intensity_arr_after_del[
                l_np_logical_and(nonzero_entry_after, intensity_arr_after_del < 1e-100)
            ] = 1e-100
            ll_events_diff = l_np_sum(
                l_np_log(intensity_arr_after_del / intensity_arr_before_del)
            )
        ll_time_diff = lower_layer.parents_events_time_to_end_expec_lst[stats_lst_id][
            chosen_index
        ]
        ll_diff = ll_events_diff + ll_time_diff
        ll_diff_dq_append(ll_diff)
        stats_diff_dq_append((intensity_arr_after_del, influenced_id_mask))
    return ll_diff_dq, stats_diff_dq


def del_event_upper_layers_virtual_stats_diff(
    this_layer, upper_layers_lst, chosen_index, children_ids_lst
):
    this_layer_id = this_layer.layer_id
    to_del_event = this_layer.real_events[chosen_index]
    stats_diff_dq = deque()
    stats_diff_dq_append = stats_diff_dq.append
    l_np_sum = np_sum
    l_np_log = np_log
    l_np_any = np_any
    ll_diff_dq = deque()
    ll_diff_dq_append = ll_diff_dq.append
    for upper_layer in upper_layers_lst:
        influenced_id_mask = upper_layer.virtual_events < to_del_event
        intensity_arr_before_del = upper_layer.virtual_intensity_arr[influenced_id_mask]

        # this layer position in upper layer lst
        upper_layer_children_ids_dict = children_ids_lst[upper_layer.layer_id]
        stats_lst_id = upper_layer_children_ids_dict[this_layer_id]

        intensity_arr_after_del = (
            intensity_arr_before_del
            - upper_layer.virtual_intensity_mat_lst[stats_lst_id][
                chosen_index, influenced_id_mask
            ]
        )
        # assert not l_np_any(intensity_arr_after_del < -1e-4)
        intensity_arr_after_del[intensity_arr_after_del < 1e-100] = 1e-100
        ll_events_diff = l_np_sum(
            l_np_log(intensity_arr_after_del / intensity_arr_before_del)
        )
        ll_time_diff = upper_layer.children_events_time_to_beginning_expec_lst[
            stats_lst_id
        ][chosen_index]
        ll_diff = ll_events_diff + ll_time_diff
        ll_diff_dq_append(ll_diff)
        stats_diff_dq_append((intensity_arr_after_del, influenced_id_mask))
    return ll_diff_dq, stats_diff_dq


def append_event_lower_layers_stats_diff(
    this_layer, lower_layers_lst, event, children_ids_lst
):
    this_layer_real_events = this_layer.real_events
    stats_diff_dq = deque()
    stats_diff_dq_append = stats_diff_dq.append
    l_np_sum = np_sum
    l_np_log = np_log
    l_np_logical_and = np_logical_and
    if len(this_layer_real_events) != 0:
        p_t_min = np_min((event, np_min(this_layer_real_events)))
    else:
        p_t_min = event
    time_diff = this_layer.end_time - event
    quasi_this_layer_to_append = quasi_layer(
        real_events=event, kernels_lst=this_layer.kernels_lst, virtual_kernels_lst=None,
    )
    this_layers_lst = [(quasi_this_layer_to_append)]
    ll_diff_dq = deque()
    ll_diff_dq_append = ll_diff_dq.append
    for lower_layer in lower_layers_lst:
        influenced_id_mask = lower_layer.real_events > event
        intensity_arr_before_append = lower_layer.intensity_arr[influenced_id_mask]

        # lower layer position in this layer
        stats_lst_ids_lst = [
            children_ids_lst[this_layer.layer_id][lower_layer.layer_id]
        ]

        intensity_mat_lst_to_append = calc_intensity_mat_lst(
            upper_layers_lst=this_layers_lst,
            events_times=lower_layer.real_events,
            stats_lst_ids_lst=stats_lst_ids_lst,
        )
        intensity_mat_to_append = intensity_mat_lst_to_append[0]
        intensity_arr_after_append = (
            intensity_arr_before_append
            + intensity_mat_to_append.flatten()[influenced_id_mask]
        )
        nonzero_entry_after = lower_layer.real_events[influenced_id_mask] > p_t_min

        intensity_arr_after_append[
            l_np_logical_and(nonzero_entry_after, intensity_arr_after_append < 1e-100)
        ] = 1e-100
        ll_events_diff = l_np_sum(
            l_np_log(intensity_arr_after_append / intensity_arr_before_append)
        )
        ll_time_diff = -this_layer.kernels_lst[stats_lst_ids_lst[0]].expectation(
            input_arr=time_diff, start_time=0
        )
        ll_diff = ll_events_diff + ll_time_diff
        ll_diff_dq_append(ll_diff)
        stats_diff_dq_append(
            (intensity_arr_after_append, influenced_id_mask, intensity_mat_to_append,)
        )
    return ll_diff_dq, stats_diff_dq


def append_event_upper_layers_virtual_stats_diff(
    this_layer, upper_layers_lst, event, parents_ids_lst
):
    if not upper_layers_lst:
        return deque(), deque()
    stats_diff_dq = deque()
    stats_diff_dq_append = stats_diff_dq.append
    quasi_this_layer_to_append = quasi_layer(
        real_events=np_array([event]),
        kernels_lst=None,
        virtual_kernels_lst=this_layer.virtual_kernels_lst,
    )
    this_layers_lst = [quasi_this_layer_to_append]
    ll_diff_dq = deque()
    ll_diff_dq_append = ll_diff_dq.append
    for upper_layer in upper_layers_lst:
        influenced_id_mask = upper_layer.virtual_events < event
        intensity_arr_before_append = upper_layer.virtual_intensity_arr[
            influenced_id_mask
        ]

        # upper layer position in this layer
        stats_lst_ids_lst = [parents_ids_lst[this_layer.layer_id][upper_layer.layer_id]]

        intensity_mat_lst_to_append = calc_virtual_intensity_mat_lst(
            lower_layers_lst=this_layers_lst,
            events_times=upper_layer.virtual_events,
            stats_lst_ids_lst=stats_lst_ids_lst,
        )
        intensity_mat_to_append = intensity_mat_lst_to_append[0]
        intensity_arr_after_append = (
            intensity_arr_before_append
            + intensity_mat_to_append.flatten()[influenced_id_mask]
        )
        # assert not np_any(intensity_arr_after_append <= 0)
        intensity_arr_after_append[intensity_arr_after_append < 1e-100] = 1e-100
        ll_events_diff = np_sum(
            np_log(intensity_arr_after_append / intensity_arr_before_append)
        )
        ll_time_diff = -this_layer.virtual_kernels_lst[
            stats_lst_ids_lst[0]
        ].expectation(input_arr=event, start_time=0)
        ll_diff = ll_events_diff + ll_time_diff
        ll_diff_dq_append(ll_diff)
        stats_diff_dq_append(
            (intensity_arr_after_append, influenced_id_mask, intensity_mat_to_append,)
        )
    return ll_diff_dq, stats_diff_dq


def swap_event_lower_layers_stats_diff(
    this_layer, lower_layers_lst, real_id, virtual_id, parents_ids_lst, children_ids_lst
):
    to_append_event = this_layer.virtual_events[virtual_id]
    to_del_event = this_layer.real_events[real_id]
    remaining_real_ids_mask = np_arange(len(this_layer.real_events)) != real_id
    remaining_real_events = this_layer.real_events[remaining_real_ids_mask]

    min_prop_time = np_min((to_append_event, to_del_event))
    if len(remaining_real_events) > 0:
        p_t_min = np_min((to_append_event, np_min(remaining_real_events),))
    else:
        p_t_min = to_append_event
    this_layer_id = this_layer.layer_id
    quasi_this_layer_to_append = quasi_layer(
        real_events=to_append_event,
        kernels_lst=this_layer.kernels_lst,
        virtual_kernels_lst=None,
    )
    this_layers_lst = [quasi_this_layer_to_append]

    stats_diff_dq = deque()
    end_time = this_layer.end_time
    ll_diff_dq = deque()
    ll_diff_dq_append = ll_diff_dq.append
    for lower_layer in lower_layers_lst:
        influenced_id_mask = lower_layer.real_events > min_prop_time
        intensity_arr_before_append = lower_layer.intensity_arr[influenced_id_mask]

        # lower layer position in this layer
        stats_lst_ids_lst = [
            children_ids_lst[this_layer.layer_id][lower_layer.layer_id]
        ]

        intensity_mat_lst_to_append = calc_intensity_mat_lst(
            upper_layers_lst=this_layers_lst,
            events_times=lower_layer.real_events,
            stats_lst_ids_lst=stats_lst_ids_lst,
        )
        intensity_mat_to_append = intensity_mat_lst_to_append[0]
        intensity_arr_after_append = (
            intensity_arr_before_append
            + intensity_mat_to_append.flatten()[influenced_id_mask]
        )
        intensity_arr_before_del = intensity_arr_after_append

        # this layer position in lower stats lst
        lower_layer_parents_ids_dict = parents_ids_lst[lower_layer.layer_id]
        stats_lst_id = lower_layer_parents_ids_dict[this_layer_id]

        intensity_arr_after_del = (
            intensity_arr_before_del
            - lower_layer.intensity_mat_lst[stats_lst_id][
                influenced_id_mask, real_id
            ].flatten()
        )
        nonzero_entry_after = lower_layer.real_events[influenced_id_mask] > p_t_min

        if np_any(np_logical_not(nonzero_entry_after)):
            ll_events_diff = -np_inf
        else:
            intensity_arr_after_del[
                np_logical_and(nonzero_entry_after, intensity_arr_after_del < 1e-100)
            ] = 1e-100
            ll_events_diff = np_sum(
                np_log(intensity_arr_after_del / intensity_arr_before_append)
            )
        ll_time_diff = (
            -this_layer.kernels_lst[stats_lst_ids_lst[0]].expectation(
                input_arr=end_time - to_append_event, start_time=0
            )
            + lower_layer.parents_events_time_to_end_expec_lst[stats_lst_id][real_id]
        )
        ll_diff = ll_events_diff + ll_time_diff
        stats_diff_dq.append(
            (intensity_arr_after_del, influenced_id_mask, intensity_mat_to_append,)
        )
        ll_diff_dq_append(ll_diff)
    return ll_diff_dq, stats_diff_dq


def swap_event_upper_layers_virtual_stats_diff(
    this_layer, upper_layers_lst, real_id, virtual_id, parents_ids_lst, children_ids_lst
):
    if not upper_layers_lst:
        return deque(), deque()
    to_append_event = this_layer.virtual_events[virtual_id]
    to_del_event = this_layer.real_events[real_id]

    max_prop_time = np_max((to_append_event, to_del_event))
    this_layer_id = this_layer.layer_id
    quasi_this_layer_to_append = quasi_layer(
        real_events=np_array([to_append_event]),
        kernels_lst=None,
        virtual_kernels_lst=this_layer.virtual_kernels_lst,
    )
    this_layers_lst = [quasi_this_layer_to_append]

    stats_diff_dq = deque()
    ll_diff_dq = deque()
    ll_diff_dq_append = ll_diff_dq.append
    for upper_layer in upper_layers_lst:
        influenced_id_mask = upper_layer.virtual_events < max_prop_time
        intensity_arr_before_append = upper_layer.virtual_intensity_arr[
            influenced_id_mask
        ]

        # upper layer position in this layer
        stats_lst_ids_lst = [parents_ids_lst[this_layer.layer_id][upper_layer.layer_id]]

        intensity_mat_lst_to_append = calc_virtual_intensity_mat_lst(
            lower_layers_lst=this_layers_lst,
            events_times=upper_layer.virtual_events,
            stats_lst_ids_lst=stats_lst_ids_lst,
        )
        intensity_mat_to_append = intensity_mat_lst_to_append[0]
        intensity_arr_after_append = (
            intensity_arr_before_append
            + intensity_mat_to_append.flatten()[influenced_id_mask]
        )
        intensity_arr_before_del = intensity_arr_after_append

        # this layer position in upper layer lst
        upper_layer_children_ids_dict = children_ids_lst[upper_layer.layer_id]
        stats_lst_id = upper_layer_children_ids_dict[this_layer_id]

        intensity_arr_after_del = (
            intensity_arr_before_del
            - upper_layer.virtual_intensity_mat_lst[stats_lst_id][
                real_id, influenced_id_mask
            ]
        )

        append_event_ll_diff = -this_layer.virtual_kernels_lst[
            stats_lst_ids_lst[0]
        ].expectation(input_arr=to_append_event, start_time=0)

        ll_time_diff = (
            upper_layer.children_events_time_to_beginning_expec_lst[stats_lst_id][
                real_id
            ]
            + append_event_ll_diff
        )
        # assert not np_any(intensity_arr_after_del < -1e-4)
        intensity_arr_after_del[intensity_arr_after_del < 1e-100] = 1e-100
        ll_events_diff = np_sum(
            np_log(intensity_arr_after_del / intensity_arr_before_append)
        )
        ll_diff = ll_events_diff + ll_time_diff
        ll_diff_dq_append(ll_diff)
        stats_diff_dq.append(
            (intensity_arr_after_del, influenced_id_mask, intensity_mat_to_append,)
        )
    return ll_diff_dq, stats_diff_dq


def swap_events_this_layer(
    this_layer,
    real_id,
    virtual_id,
    intensity_arr_to_append,
    virtual_intensity_arr_to_append,
    log_real_intensity_to_add,
    log_virtual_intensity_to_add,
    intensity_mat_lst_to_append,
    virtual_intensity_mat_lst_to_append,
):
    this_layer_real_events = this_layer.real_events
    this_layer_virtual_events = this_layer.virtual_events
    remaining_real_ids_mask = np_ones(len(this_layer_real_events), dtype=bool)
    remaining_virtual_ids_mask = np_ones(len(this_layer_virtual_events), dtype=bool)
    remaining_real_ids_mask[real_id] = False
    remaining_virtual_ids_mask[virtual_id] = False
    to_append_real_event = this_layer.virtual_events[virtual_id]
    to_append_virtual_event = this_layer.real_events[real_id]
    this_layer.virtual_events = this_layer.virtual_events[remaining_virtual_ids_mask]
    this_layer.virtual_events = np_append(
        this_layer.virtual_events, to_append_virtual_event
    )
    this_layer.real_events = this_layer.real_events[remaining_real_ids_mask]
    this_layer.real_events = np_append(this_layer.real_events, to_append_real_event)
    if "virtual_kernels_param_lst" in this_layer.__dict__:
        this_layer.loglikelihood -= np_log(this_layer.intensity_arr[real_id])
        this_layer.loglikelihood += log_real_intensity_to_add

        # TODO need to check
        this_layer_intensity_mat_lst = this_layer.intensity_mat_lst
        for count, intensity_mat in enumerate(this_layer_intensity_mat_lst):
            intensity_mat_temp = intensity_mat[remaining_real_ids_mask]
            to_append_real_mat = intensity_mat_lst_to_append[count]
            intensity_mat_temp = np_concatenate(
                (intensity_mat_temp, to_append_real_mat), axis=0
            )
            this_layer_intensity_mat_lst[count] = intensity_mat_temp
        this_layer.intensity_arr = this_layer.intensity_arr[remaining_real_ids_mask]
        this_layer.intensity_arr = np_append(
            this_layer.intensity_arr, intensity_arr_to_append,
        )

    this_layer.virtual_loglikelihood += log_virtual_intensity_to_add
    this_layer.virtual_loglikelihood -= np_log(
        this_layer.virtual_intensity_arr[virtual_id]
    )

    # TODO need to check
    this_layer_virtual_intensity_mat_lst = this_layer.virtual_intensity_mat_lst
    for (count, virtual_intensity_mat,) in enumerate(
        this_layer_virtual_intensity_mat_lst
    ):
        virtual_intensity_mat_temp = virtual_intensity_mat[
            :, remaining_virtual_ids_mask
        ]
        to_append_virtual_mat = virtual_intensity_mat_lst_to_append[count]
        virtual_intensity_mat_temp = np_concatenate(
            (virtual_intensity_mat_temp, to_append_virtual_mat,), axis=1,
        )
        this_layer_virtual_intensity_mat_lst[count] = virtual_intensity_mat_temp
    this_layer.virtual_intensity_arr = this_layer.virtual_intensity_arr[
        remaining_virtual_ids_mask
    ]
    this_layer.virtual_intensity_arr = np_append(
        this_layer.virtual_intensity_arr, virtual_intensity_arr_to_append
    )


def flip_real_event_this_layer(
    this_layer,
    chosen_index,
    virtual_intensity_mat_lst_to_append,
    virtual_intensity_arr_to_append,
    log_virtual_intensity_arr_to_append,
):
    this_layer_real_events = this_layer.real_events
    remaining_ids_mask = np_ones(len(this_layer_real_events), dtype=bool)
    remaining_ids_mask[chosen_index] = False
    to_append_virtual_event = this_layer_real_events[chosen_index]
    this_layer.virtual_events = np_append(
        this_layer.virtual_events, to_append_virtual_event
    )
    this_layer.real_events = this_layer_real_events[remaining_ids_mask]
    # if isinstance(this_layer, TopLayerEvents):
    if "virtual_kernels_param_lst" not in this_layer.__dict__:
        this_layer.loglikelihood -= np_log(this_layer.base_rate)
    else:
        this_layer.loglikelihood -= np_log(this_layer.intensity_arr[chosen_index])
        # TODO need to check
        this_layer_intensity_mat_lst = this_layer.intensity_mat_lst
        for count, intensity_mat in enumerate(this_layer_intensity_mat_lst):
            this_layer_intensity_mat_lst[count] = intensity_mat[remaining_ids_mask]
        this_layer.intensity_arr = this_layer.intensity_arr[remaining_ids_mask]

    this_layer_virtual_intensity_mat_lst = this_layer.virtual_intensity_mat_lst
    for count, virtual_intensity_mat in enumerate(this_layer_virtual_intensity_mat_lst):
        to_append_virtual_intensity_mat = virtual_intensity_mat_lst_to_append[count]
        virtual_intensity_mat = np_concatenate(
            (virtual_intensity_mat, to_append_virtual_intensity_mat), axis=1,
        )
        this_layer_virtual_intensity_mat_lst[count] = virtual_intensity_mat
    this_layer.virtual_intensity_arr = np_append(
        this_layer.virtual_intensity_arr, virtual_intensity_arr_to_append
    )
    this_layer.virtual_loglikelihood += log_virtual_intensity_arr_to_append


def flip_virtual_event_this_layer(
    this_layer,
    chosen_index,
    intensity_mat_lst_to_append,
    intensity_arr_to_append,
    log_intensity_arr_to_append,
):
    this_layer_virtual_events = this_layer.virtual_events
    remaining_ids_mask = np_ones(len(this_layer_virtual_events), dtype=bool)
    remaining_ids_mask[chosen_index] = False
    to_append_real_event = this_layer_virtual_events[chosen_index]
    this_layer.real_events = np_append(this_layer.real_events, to_append_real_event)
    this_layer.virtual_events = this_layer_virtual_events[remaining_ids_mask]
    if "virtual_kernels_param_lst" not in this_layer.__dict__:
        this_layer.loglikelihood += np_log(this_layer.base_rate)
    else:
        this_layer_intensity_mat_lst = this_layer.intensity_mat_lst
        for count, intensity_mat in enumerate(this_layer_intensity_mat_lst):
            to_append_intensity_mat = intensity_mat_lst_to_append[count]
            intensity_mat = np_concatenate(
                (intensity_mat, to_append_intensity_mat), axis=0
            )
            this_layer_intensity_mat_lst[count] = intensity_mat
        this_layer.loglikelihood += log_intensity_arr_to_append
        this_layer.intensity_arr = np_append(
            this_layer.intensity_arr, intensity_arr_to_append
        )

    this_layer.virtual_loglikelihood -= np_log(
        this_layer.virtual_intensity_arr[chosen_index]
    )
    # TODO need to check
    this_layer_virtual_intensity_mat_lst = this_layer.virtual_intensity_mat_lst
    for count, virtual_intensity_mat in enumerate(this_layer_virtual_intensity_mat_lst):
        this_layer_virtual_intensity_mat_lst[count] = virtual_intensity_mat[
            :, remaining_ids_mask
        ]
    this_layer.virtual_intensity_arr = this_layer.virtual_intensity_arr[
        remaining_ids_mask
    ]


def change_real_event_neighbor_layers(
    change_type,
    chosen_index,
    dpp_layers_events,
    this_layer_id,
    to_append_event,
    ll_diff_dq,
    stats_diff_dq,
    virtual_ll_diff_dq,
    virtual_stats_diff_dq,
    swap_real_id,
    swap_virtual_id,
    parents_ids_lst,
    children_ids_lst,
):
    remaining_ids_mask = (
        np_arange(len(dpp_layers_events.layers[this_layer_id].real_events))
        != chosen_index
    )
    if change_type == "swap":
        remaining_real_ids_mask = (
            np_arange(len(dpp_layers_events.layers[this_layer_id].real_events))
            != swap_real_id
        )
    stats_diff_dq_popleft = stats_diff_dq.popleft
    layers = dpp_layers_events.layers
    this_layer = layers[this_layer_id]
    end_time = this_layer.end_time
    if change_type == "del":
        for c in children_ids_lst[this_layer_id].keys():
            ll_diff = ll_diff_dq.popleft()
            intensity_arr, influenced_id_mask = stats_diff_dq_popleft()
            c_layer = layers[c]
            c_layer.intensity_arr[influenced_id_mask] = intensity_arr
            c_layer_intensity_mat_lst = c_layer.intensity_mat_lst
            c_layer_parents_events_time_to_end_expec_lst = (
                c_layer.parents_events_time_to_end_expec_lst
            )
            c_layer.loglikelihood += ll_diff

            # this layer position in lower layer list
            lower_layer_parents_ids_dict = parents_ids_lst[c]
            stats_lst_id = lower_layer_parents_ids_dict[this_layer_id]

            c_layer_intensity_mat_lst[stats_lst_id] = c_layer_intensity_mat_lst[
                stats_lst_id
            ][:, remaining_ids_mask]
            c_layer_parents_events_time_to_end_expec_lst[
                stats_lst_id
            ] = c_layer_parents_events_time_to_end_expec_lst[stats_lst_id][
                remaining_ids_mask
            ]
    elif change_type == "append":
        for c in children_ids_lst[this_layer_id].keys():
            ll_diff = ll_diff_dq.popleft()
            (
                intensity_arr,
                influenced_id_mask,
                intensity_mat_to_append,
            ) = stats_diff_dq_popleft()
            c_layer = layers[c]
            c_layer.intensity_arr[influenced_id_mask] = intensity_arr
            c_layer_intensity_mat_lst = c_layer.intensity_mat_lst

            # this layer position in lower layer list
            lower_layer_parents_ids_dict = parents_ids_lst[c]
            stats_lst_id = lower_layer_parents_ids_dict[this_layer_id]

            # TODO need to check
            c_layer_parents_events_time_to_end_expec_lst = (
                c_layer.parents_events_time_to_end_expec_lst
            )
            c_layer.loglikelihood += ll_diff

            c_layer_intensity_mat_lst[stats_lst_id] = np_concatenate(
                (c_layer_intensity_mat_lst[stats_lst_id], intensity_mat_to_append,),
                axis=1,
            )

            # lower layer position in this layer
            lower_position_stats_lst_id = children_ids_lst[this_layer_id][c]

            c_layer_parents_events_time_to_end_expec_lst[stats_lst_id] = np_append(
                c_layer_parents_events_time_to_end_expec_lst[stats_lst_id],
                this_layer.kernels_lst[lower_position_stats_lst_id].expectation(
                    input_arr=end_time - to_append_event, start_time=0
                ),
            )
    elif change_type == "swap":
        for c in children_ids_lst[this_layer_id].keys():
            ll_diff = ll_diff_dq.popleft()
            (
                intensity_arr,
                influenced_id_mask,
                intensity_mat_to_append,
            ) = stats_diff_dq_popleft()
            c_layer = layers[c]
            c_layer.intensity_arr[influenced_id_mask] = intensity_arr
            c_layer_intensity_mat_lst = c_layer.intensity_mat_lst
            c_layer_parents_events_time_to_end_expec_lst = (
                c_layer.parents_events_time_to_end_expec_lst
            )
            c_layer.loglikelihood += ll_diff

            # this layer position in lower layer list
            lower_layer_parents_ids_dict = parents_ids_lst[c]
            stats_lst_id = lower_layer_parents_ids_dict[this_layer_id]

            c_layer_intensity_mat_lst[stats_lst_id] = c_layer_intensity_mat_lst[
                stats_lst_id
            ][:, remaining_real_ids_mask]
            c_layer_parents_events_time_to_end_expec_lst[
                stats_lst_id
            ] = c_layer_parents_events_time_to_end_expec_lst[stats_lst_id][
                remaining_real_ids_mask
            ]
            c_layer_intensity_mat_lst[stats_lst_id] = np_concatenate(
                (c_layer_intensity_mat_lst[stats_lst_id], intensity_mat_to_append,),
                axis=1,
            )

            # lower layer position in this layer
            lower_position_stats_lst_id = children_ids_lst[this_layer_id][c]

            c_layer_parents_events_time_to_end_expec_lst[stats_lst_id] = np_append(
                c_layer_parents_events_time_to_end_expec_lst[stats_lst_id],
                this_layer.kernels_lst[lower_position_stats_lst_id].expectation(
                    input_arr=end_time - to_append_event, start_time=0
                ),
            )

    virtual_stats_diff_dq_popleft = virtual_stats_diff_dq.popleft
    if layers[this_layer_id].virtual_events is not None:
        if change_type == "del":
            for u in parents_ids_lst[this_layer_id].keys():
                virtual_ll_diff = virtual_ll_diff_dq.popleft()
                (
                    virtual_intensity_arr,
                    virtual_influenced_id_mask,
                ) = virtual_stats_diff_dq_popleft()
                u_layer = layers[u]
                u_layer_virtual_intensity_mat_lst = u_layer.virtual_intensity_mat_lst
                u_layer_children_events_time_to_beginning_expec_lst = (
                    u_layer.children_events_time_to_beginning_expec_lst
                )
                u_layer.virtual_loglikelihood += virtual_ll_diff
                u_layer.virtual_intensity_arr[
                    virtual_influenced_id_mask
                ] = virtual_intensity_arr

                # this layer position in upper layer list
                upper_layer_children_ids_dict = children_ids_lst[u]
                stats_lst_id = upper_layer_children_ids_dict[this_layer_id]

                u_layer_virtual_intensity_mat_lst[
                    stats_lst_id
                ] = u_layer_virtual_intensity_mat_lst[stats_lst_id][
                    remaining_ids_mask, :
                ]
                u_layer_children_events_time_to_beginning_expec_lst[
                    stats_lst_id
                ] = u_layer_children_events_time_to_beginning_expec_lst[stats_lst_id][
                    remaining_ids_mask
                ]
        elif change_type == "append":
            for u in parents_ids_lst[this_layer_id].keys():
                virtual_ll_diff = virtual_ll_diff_dq.popleft()
                (
                    virtual_intensity_arr,
                    virtual_influenced_id_mask,
                    virtual_intensity_mat_to_append,
                ) = virtual_stats_diff_dq_popleft()
                u_layer = layers[u]
                u_layer_virtual_intensity_mat_lst = u_layer.virtual_intensity_mat_lst
                u_layer_children_events_time_to_beginning_expec_lst = (
                    u_layer.children_events_time_to_beginning_expec_lst
                )
                u_layer.virtual_loglikelihood += virtual_ll_diff
                u_layer.virtual_intensity_arr[
                    virtual_influenced_id_mask
                ] = virtual_intensity_arr

                # this layer position in upper layer list
                upper_layer_children_ids_dict = children_ids_lst[u]
                stats_lst_id = upper_layer_children_ids_dict[this_layer_id]

                u_layer.virtual_intensity_mat_lst[stats_lst_id] = np_concatenate(
                    (
                        u_layer_virtual_intensity_mat_lst[stats_lst_id],
                        virtual_intensity_mat_to_append,
                    ),
                    axis=0,
                )

                # upper layer position in this layer
                upper_stats_lst_id = parents_ids_lst[this_layer.layer_id][u_layer.layer_id]

                u_layer_children_events_time_to_beginning_expec_lst[
                    stats_lst_id
                ] = np_append(
                    u_layer_children_events_time_to_beginning_expec_lst[stats_lst_id],
                    this_layer.virtual_kernels_lst[upper_stats_lst_id].expectation(
                        input_arr=to_append_event, start_time=0
                    ),
                )
        elif change_type == "swap":
            for u in parents_ids_lst[this_layer_id].keys():
                virtual_ll_diff = virtual_ll_diff_dq.popleft()
                (
                    virtual_intensity_arr,
                    virtual_influenced_id_mask,
                    virtual_intensity_mat_to_append,
                ) = virtual_stats_diff_dq_popleft()
                u_layer = layers[u]
                u_layer_virtual_intensity_mat_lst = u_layer.virtual_intensity_mat_lst
                u_layer_children_events_time_to_beginning_expec_lst = (
                    u_layer.children_events_time_to_beginning_expec_lst
                )
                u_layer.virtual_loglikelihood += virtual_ll_diff
                u_layer.virtual_intensity_arr[
                    virtual_influenced_id_mask
                ] = virtual_intensity_arr

                # this layer position in upper layer list
                upper_layer_children_ids_dict = children_ids_lst[u]
                stats_lst_id = upper_layer_children_ids_dict[this_layer_id]

                u_layer_virtual_intensity_mat_lst[
                    stats_lst_id
                ] = u_layer_virtual_intensity_mat_lst[stats_lst_id][
                    remaining_real_ids_mask, :
                ]
                u_layer_virtual_intensity_mat_lst[stats_lst_id] = np_concatenate(
                    (
                        u_layer_virtual_intensity_mat_lst[stats_lst_id],
                        virtual_intensity_mat_to_append,
                    ),
                    axis=0,
                )
                u_layer_children_events_time_to_beginning_expec_lst[
                    stats_lst_id
                ] = u_layer_children_events_time_to_beginning_expec_lst[stats_lst_id][
                    remaining_real_ids_mask
                ]

                # upper layer position in this layer
                upper_stats_lst_id = parents_ids_lst[this_layer.layer_id][u_layer.layer_id]

                u_layer_children_events_time_to_beginning_expec_lst[
                    stats_lst_id
                ] = np_append(
                    u_layer_children_events_time_to_beginning_expec_lst[stats_lst_id],
                    this_layer.virtual_kernels_lst[upper_stats_lst_id].expectation(
                        input_arr=to_append_event, start_time=0
                    ),
                )


def neighbor_layers_stats_diff_after_moving_event(
    dpp_layers_events,
    this_layer_id,
    chosen_index,
    event_to_append,
    move_type,
    real_id,
    virtual_id,
):
    layers = dpp_layers_events.layers
    children_ids_lst = dpp_layers_events.children_ids_lst
    parents_ids_lst = dpp_layers_events.parents_ids_lst
    this_layer = layers[this_layer_id]
    lower_layers_lst = [layers[c] for c in children_ids_lst[this_layer_id].keys()]
    upper_layers_lst = [layers[p] for p in parents_ids_lst[this_layer_id].keys()]
    if move_type == "del":
        ll_diff_dq, stats_diff_dq = del_event_lower_layers_stats_diff(
            this_layer, lower_layers_lst, chosen_index, parents_ids_lst=parents_ids_lst
        )
    elif move_type == "append":
        ll_diff_dq, stats_diff_dq = append_event_lower_layers_stats_diff(
            this_layer,
            lower_layers_lst,
            event_to_append,
            children_ids_lst=children_ids_lst,
        )
    elif move_type == "swap":
        ll_diff_dq, stats_diff_dq = swap_event_lower_layers_stats_diff(
            this_layer,
            lower_layers_lst,
            real_id,
            virtual_id,
            parents_ids_lst=parents_ids_lst,
            children_ids_lst=children_ids_lst,
        )

    if this_layer.virtual_events is not None:
        if move_type == "del":
            (
                virtual_ll_diff_dq,
                virtual_stats_diff_dq,
            ) = del_event_upper_layers_virtual_stats_diff(
                this_layer,
                upper_layers_lst,
                chosen_index,
                children_ids_lst=children_ids_lst,
            )
        elif move_type == "append":
            (
                virtual_ll_diff_dq,
                virtual_stats_diff_dq,
            ) = append_event_upper_layers_virtual_stats_diff(
                this_layer,
                upper_layers_lst,
                event_to_append,
                parents_ids_lst=parents_ids_lst,
            )
        elif move_type == "swap":
            (
                virtual_ll_diff_dq,
                virtual_stats_diff_dq,
            ) = swap_event_upper_layers_virtual_stats_diff(
                this_layer,
                upper_layers_lst,
                real_id,
                virtual_id,
                parents_ids_lst=parents_ids_lst,
                children_ids_lst=children_ids_lst,
            )
    return ll_diff_dq, virtual_ll_diff_dq, stats_diff_dq, virtual_stats_diff_dq
