import numpy as np
from utils import (
    flip_real_event_this_layer,
    flip_virtual_event_this_layer,
    change_real_event_neighbor_layers,
    neighbor_layers_stats_diff_after_moving_event,
    swap_events_this_layer,
)
from numpy.random import default_rng
from DPPLayers import (
    calc_virtual_intensity_mat_lst,
    calc_virtual_intensity_arr,
    sample_from_lower_layers_to_upper,
    calc_intensity_mat_lst,
    calc_intensity_arr,
)
from collections import deque

np_min = np.min
np_sum = np.sum
np_log = np.log
np_inf = np.inf
np_array = np.array
np_any = np.any


class PriorSampler:
    def __init__(
        self, dpp_layers_events, virtual=False, seed=None
    ):
        self.dpp_layers_events = dpp_layers_events
        self.virtual = virtual
        self.seed = seed
        if seed is not None:
            self.rng = default_rng(seed)

    def sample(self, rng=None):
        if rng is None:
            rng = self.rng
        dpp_layers_events = self.dpp_layers_events
        parents_ids_lst = dpp_layers_events.parents_ids_lst
        children_ids_lst = dpp_layers_events.children_ids_lst
        layers = dpp_layers_events.layers
        virtual = self.virtual
        for layer_id in dpp_layers_events.evidences_ids_set:
            to_sample_layers_ids_keys = parents_ids_lst[layer_id].keys()
            for to_sample_id in to_sample_layers_ids_keys:
                children_ids_keys = children_ids_lst[to_sample_id].keys()
                lower_layers_lst = [layers[l_id] for l_id in children_ids_keys]
                lower_layers_lst_not_done = [
                    layer.real_events is None for layer in lower_layers_lst
                ]

                if (
                    not any(lower_layers_lst_not_done)
                    and layers[to_sample_id].real_events is None
                ):
                    stats_lst_ids_lst = [
                        parents_ids_lst[c_id][to_sample_id]
                        for c_id in children_ids_keys
                    ]
                    layers[to_sample_id].prior_sample(
                        lower_layers_lst=lower_layers_lst,
                        virtual=virtual,
                        rng=rng,
                        stats_lst_ids_lst=stats_lst_ids_lst,
                    )
                self.sample_parent(to_sample_id, rng)
        for layer in dpp_layers_events.layers:

            dpp_layers_events.set_layer_events(
                layer.layer_id,
                real_events=None,
                virtual_events=None,
                this_layer_virtual_ll=layer.virtual_loglikelihood,
            )

    def sample_parent(self, layer_id, rng):
        dpp_layers_events = self.dpp_layers_events
        parents_ids_lst = dpp_layers_events.parents_ids_lst
        children_ids_lst = dpp_layers_events.children_ids_lst
        layers = dpp_layers_events.layers
        virtual = self.virtual
        if parents_ids_lst[layer_id]:
            for p in parents_ids_lst[layer_id]:
                children_ids_keys = children_ids_lst[p].keys()
                lower_layers_lst = [layers[l_id] for l_id in children_ids_keys]

                lower_layers_lst_not_done = [
                    layer.real_events is None for layer in lower_layers_lst
                ]

                p_layer = layers[p]
                if not any(lower_layers_lst_not_done) and p_layer.real_events is None:
                    stats_lst_ids_lst = [
                        parents_ids_lst[c_id][p] for c_id in children_ids_keys
                    ]

                    p_layer.prior_sample(
                        lower_layers_lst=lower_layers_lst,
                        virtual=virtual,
                        rng=rng,
                        stats_lst_ids_lst=stats_lst_ids_lst,
                    )
                    self.sample_parent(p, rng)


class PosteriorSampler:
    def __init__(
        self,
        dpp_layers_events,
        rng,
        ex_id,
        prior_sample=None,
        prior_virtual_sample=None,
        virtual=False,
        virtual_resample_prob=0.2,
        swap_prob=0.2,
        random_sample_layer=False,
        check_parents=False,
    ):
        self.dpp_layers_events = dpp_layers_events
        self.virtual_resample_prob = virtual_resample_prob
        self.swap_prob = swap_prob

        if prior_sample is None:
            self.prior_sampler = PriorSampler(
                dpp_layers_events=self.dpp_layers_events,
                virtual=virtual,
            )
        self.prior_sample = prior_sample
        self.prior_virtual_sample = prior_virtual_sample
        self.MCMCSampler = None

        self.rng = rng
        self.MCMCLayers_ids_lst = None

        self.ex_id = ex_id
        self.random_sample_layer = random_sample_layer
        self.check_parents = check_parents

    def check_parents_valid(self, c_layer_id):
        dpp_layers_events = self.dpp_layers_events
        layers = dpp_layers_events.layers
        children_ids_lst = dpp_layers_events.children_ids_lst
        this_layer = layers[c_layer_id]
        parents_ids_lst = dpp_layers_events.parents_ids_lst
        parents_ids_dict = parents_ids_lst[c_layer_id]
        if not parents_ids_dict:
            return
        for u_layer_id in parents_ids_dict.keys():
            u_layer = dpp_layers_events.layers[u_layer_id]
            if this_layer.base_rate == 0:
                if len(this_layer.real_events) != 0:
                    this_min = np_min(this_layer.real_events)
                else:
                    this_min = None
                if len(u_layer.real_events) != 0:
                    u_min = np_min(u_layer.real_events)
                else:
                    u_min = np_inf
                if this_min is not None and u_min >= this_min:
                    children_ids_keys = children_ids_lst[u_layer_id].keys()
                    lower_layers_lst = [layers[l_id] for l_id in children_ids_keys]

                    stats_lst_ids_lst = [
                        parents_ids_lst[c_id][u_layer_id]
                        for c_id in children_ids_keys
                    ]
                    layers[u_layer_id].prior_sample(
                        lower_layers_lst=lower_layers_lst,
                        virtual=True,
                        rng=self.rng,
                        stats_lst_ids_lst=stats_lst_ids_lst,
                    )
                    self.check_parents_valid(u_layer_id)


    def virtual_mixing(self, burn_in_steps):
        dpp_layers_events = self.dpp_layers_events
        layers = dpp_layers_events.layers
        evidences_ids_set = dpp_layers_events.evidences_ids_set
        prior_sample = self.prior_sample
        rng = self.rng
        virtual_resample_prob = self.virtual_resample_prob
        swap_prob = self.swap_prob
        layers_ids_lst = [layer.layer_id for layer in layers]
        if prior_sample is None:
            self.prior_sampler.sample(rng=rng)
            # for layer_id in self.dpp_layers_events.layers:
            #     if len(self.dpp_layers_events.children_ids_dict[layer_id]) == 0:
            #         self.check_ll_valid(layer_id)

        else:

            for layer_id in layers_ids_lst:
                if layer_id not in evidences_ids_set:
                    dpp_layers_events.set_layer_events(
                        layer_id,
                        real_events=self.prior_sample[layer_id],
                        virtual_events=self.prior_virtual_sample[layer_id],
                        this_layer_ll=-np_inf,
                        this_layer_virtual_ll=-np_inf,
                    )
            if self.check_parents:
                for layer_id in dpp_layers_events.evidences_ids_set:
                    self.check_parents_valid(layer_id)
            
            for layer_id in layers_ids_lst:
                dpp_layers_events.set_layer_events(
                    layer_id, real_events=None, virtual_events=None
                )
        self.MCMCSampler = MCMCSampler(dpp_layers_events, 1, self.ex_id)
        self.MCMCLayers_ids_lst = [
            l_id for l_id in reversed(layers_ids_lst) if l_id not in evidences_ids_set
        ]
        self.MCMCSampler.virtualMCMCLayers(
            self.MCMCLayers_ids_lst,
            virtual_resample_prob=virtual_resample_prob,
            swap_prob=swap_prob,
            rng=rng,
            sample_intervals=burn_in_steps,
            num_of_samples=1,
            burn_in=True,
            random_sample_layer=self.random_sample_layer,
            predict=False,
        )


class MCMCSampler:
    def __init__(self, dpp_layers_events, burn_in_steps, ex_id):
        self.dpp_layers_events = dpp_layers_events
        self.burn_in_steps = burn_in_steps
        self.jump_attempts = 0
        self.jump_acceptance = 0

        self.tot_joint_ll = 0
        self.tot_joint_virtual_ll = 0
        self.mixed_joint_ll = 0
        self.mixed_joint_virtual_ll = 0
        self.joint_ll_count = 0
        self.mixed_ll_count = 0

        self.ex_id = ex_id

    def virtualMCMCLayers(
        self,
        layers_ids_lst,
        virtual_resample_prob,
        swap_prob,
        rng,
        sample_intervals,
        num_of_samples,
        burn_in,
        random_sample_layer,
        predict,
    ):
        dpp_layers_events = self.dpp_layers_events
        layers = dpp_layers_events.layers
        rng_uniform = rng.uniform
        children_ids_lst = dpp_layers_events.children_ids_lst
        parents_ids_lst = dpp_layers_events.parents_ids_lst
        l_sample_from_lower_layers_to_upper = sample_from_lower_layers_to_upper
        fast_virtualMCMCAccept = self.fast_virtualMCMCAccept
        if not burn_in:
            samples_dict = {}
            virtual_samples_dict = {}
            for layer_id in layers_ids_lst:
                samples_dict[layer_id] = []
                virtual_samples_dict[layer_id] = []
        if predict:
            first_event_dict_dq = deque([])
            first_event_dict_dq_append = first_event_dict_dq.append
        if random_sample_layer:
            rng.shuffle(layers_ids_lst)
        joint_ll = self.joint_ll
        joint_virtual_ll = self.joint_virtual_ll
        tot_joint_ll = self.tot_joint_ll
        tot_joint_virtual_ll = self.tot_joint_virtual_ll
        mixed_joint_ll = self.mixed_joint_ll
        mixed_joint_virtual_ll = self.mixed_joint_virtual_ll
        mixed_ll_count = self.mixed_ll_count
        joint_ll_count = self.joint_ll_count
        for i in range(num_of_samples):
            for j in range(sample_intervals):
                for layer_index in layers_ids_lst:
                    this_layer = layers[layer_index]
                    choose_prob = rng_uniform()
                    children_ids_keys = children_ids_lst[layer_index].keys()
                    lower_layers_lst = [layers[c_id] for c_id in children_ids_keys]

                    stats_lst_ids_lst = [
                        parents_ids_lst[c_id][layer_index] for c_id in children_ids_keys
                    ]

                    if choose_prob < virtual_resample_prob:
                        (
                            this_layer.virtual_events,
                            children_events_time_to_beginning_expec_lst,
                            virtual_intensity_mat_lst,
                            virtual_intensity_arr,
                        ) = l_sample_from_lower_layers_to_upper(
                            lower_layers_lst=lower_layers_lst,
                            base_rate=this_layer.virtual_base_rate,
                            rng=rng,
                            real_prior=False,
                            start_time=None,
                            end_time=None,
                            stats_lst_ids_lst=stats_lst_ids_lst,
                        )
                        this_layer.children_events_time_to_beginning_expec_lst = (
                            children_events_time_to_beginning_expec_lst
                        )
                        this_layer.virtual_intensity_mat_lst = virtual_intensity_mat_lst
                        this_layer.virtual_intensity_arr = virtual_intensity_arr
                        this_layer.virtual_loglikelihood = this_layer.virtual_ll(
                            children_events_time_to_beginning_expec_lst,
                            virtual_intensity_arr,
                            lower_layers_lst,
                            stats_lst_ids_lst,
                        )
                        # print('virtual resample i = ', i, ', j = ', j, ', l_id = ', layer_index)
                        # self.check_ll(parents_ids_lst, children_ids_lst, layers)

                    else:
                        real_events_num = len(this_layer.real_events)
                        virtual_events_num = len(this_layer.virtual_events)
                        this_layer.real_events_num += real_events_num
                        this_layer.virtual_events_num += virtual_events_num
                        events_tot_num = real_events_num + virtual_events_num
                        if events_tot_num > 0:
                            if choose_prob < virtual_resample_prob + swap_prob:
                                if real_events_num > 0 and virtual_events_num > 0:
                                    # print('before swap i = ', i, ', j = ', j, ', l_id = ', layer_index)
                                    # self.check_ll(parents_ids_lst, children_ids_lst, layers)
                                    fast_virtualMCMCAccept(
                                        layer_index,
                                        rng=rng,
                                        swap=True,
                                        parents_ids_lst=parents_ids_lst,
                                        children_ids_lst=children_ids_lst,
                                    )
                                    # print('swap i = ', i, ', j = ', j, ', l_id = ', layer_index)
                                    # self.check_ll(parents_ids_lst, children_ids_lst, layers)
                            else:
                                # print('before move no swap i = ', i, ', j = ', j, ', l_id = ', layer_index)
                                # self.check_ll(parents_ids_lst, children_ids_lst, layers)
                                fast_virtualMCMCAccept(
                                    layer_index,
                                    rng=rng,
                                    swap=False,
                                    parents_ids_lst=parents_ids_lst,
                                    children_ids_lst=children_ids_lst,
                                )
                                # print('no swap i = ', i, ', j = ', j, ', l_id = ', layer_index)
                                # self.check_ll(parents_ids_lst, children_ids_lst, layers)

                ll = joint_ll()
                joint_ll_count += 1
                virtual_ll = joint_virtual_ll()
                sum_ll = sum(ll)
                # sum_virtual_ll = np_array(virtual_ll)
                sum_virtual_ll = sum(virtual_ll)
                tot_joint_ll += sum_ll
                tot_joint_virtual_ll += sum_virtual_ll

            # check ll for the last sample
            # if i == num_of_samples - 1:
            #     self.check_ll(parents_ids_lst, children_ids_lst, layers)
            #     # tot_real_events_num = 0
            #     # tot_virtual_events_num = 0
            #     # for layer in layers:
            #     #     tot_real_events_num += layer.real_events_num
            #     #     tot_virtual_events_num += layer.virtual_events_num
            #     # print(
            #     #     "tot real events num = ",
            #     #     tot_real_events_num,
            #     #     "tot virtual events num = ",
            #     #     tot_virtual_events_num,
            #     # )

            if not burn_in:
                ll = joint_ll()
                virtual_ll = joint_virtual_ll()
                sum_ll = sum(ll)
                # sum_virtual_ll = np_array(virtual_ll)
                sum_virtual_ll = sum(virtual_ll)
                tot_joint_ll += sum_ll
                mixed_joint_ll += sum_ll
                tot_joint_virtual_ll += sum_virtual_ll
                mixed_joint_virtual_ll += sum_virtual_ll
                mixed_ll_count += 1
                joint_ll_count += 1
                for layer_id in layers_ids_lst:
                    samples_dict[layer_id].append(layers[layer_id].real_events)
                    virtual_samples_dict[layer_id].append(
                        layers[layer_id].virtual_events
                    )
                if predict:
                    first_event_dict_dq_append(
                        dpp_layers_events.sample_first_event(rng)
                    )

        self.tot_joint_ll = tot_joint_ll
        self.tot_joint_virtual_ll = tot_joint_virtual_ll
        self.mixed_joint_ll = mixed_joint_ll
        self.mixed_joint_virtual_ll = mixed_joint_virtual_ll
        self.mixed_ll_count = mixed_ll_count
        self.joint_ll_count = joint_ll_count
        if not burn_in and not predict:
            return samples_dict, virtual_samples_dict
        elif not burn_in and predict:
            return samples_dict, virtual_samples_dict, first_event_dict_dq

    def check_ll(self, parents_ids_lst, children_ids_lst, layers):
        ll = self.joint_ll()
        virtual_ll = self.joint_virtual_ll()
        sum_ll = sum(ll)
        sum_virtual_ll = sum(virtual_ll)
        # calc ll from intensity_mat_lst
        intensity_mat_ll = 0
        for layer in self.dpp_layers_events.layers:
            parents_ids_keys = parents_ids_lst[layer.layer_id].keys()
            if len(parents_ids_keys) != 0:
                upper_layers_lst = [layers[layer_id] for layer_id in parents_ids_keys]
                p_min = np_min(
                    [
                        np_min(upper_layer.real_events)
                        if len(upper_layer.real_events) != 0
                        else np_inf
                        for upper_layer in upper_layers_lst
                    ]
                )
                intensity_arr = calc_intensity_arr(
                    p_min, layer.intensity_mat_lst, layer.real_events, layer.base_rate,
                )
                intensity_mat_ll += layer.ll(None, intensity_arr)
            else:
                intensity_mat_ll += layer.ll()

        # calc ll from intensity arr
        intensity_arr_ll = 0
        for layer in self.dpp_layers_events.layers:
            if "virtual_kernels_param_lst" in layer.__dict__:
                intensity_arr_ll += layer.ll(None, layer.intensity_arr)
            else:
                intensity_arr_ll += layer.ll()

        if (
            np.abs(intensity_mat_ll - intensity_arr_ll) > 0.1
            or np.abs(sum_ll - intensity_arr_ll) > 0.1
        ):
            print(
                "ex_id = ",
                self.ex_id,
                ", intensity mat ll = ",
                intensity_mat_ll,
                ", intensity arr ll = ",
                intensity_arr_ll,
                ", ll = ",
                sum_ll,
            )

        # calc ll from virtual_intensity_mat_lst
        intensity_mat_virtual_ll = []
        for layer in self.dpp_layers_events.layers:
            if layer.layer_id not in self.dpp_layers_events.evidences_ids_set:
                children_ids_keys = children_ids_lst[layer.layer_id].keys()
                lower_layers_lst = [layers[layer_id] for layer_id in children_ids_keys]

                stats_lst_ids_lst = [
                    parents_ids_lst[c_id][layer.layer_id] for c_id in children_ids_keys
                ]

                intensity_arr = calc_virtual_intensity_arr(
                    layer.virtual_intensity_mat_lst,
                    layer.virtual_events,
                    layer.virtual_base_rate,
                    lower_layers_lst,
                    stats_lst_ids_lst,
                )

                intensity_mat_virtual_ll.append(
                    layer.virtual_ll(
                        None, intensity_arr, lower_layers_lst, stats_lst_ids_lst
                    )
                )

        # calc ll from intensity arr
        intensity_arr_virtual_ll = []
        for layer in self.dpp_layers_events.layers:
            if layer.layer_id not in self.dpp_layers_events.evidences_ids_set:
                children_ids_keys = children_ids_lst[layer.layer_id].keys()
                lower_layers_lst = [layers[layer_id] for layer_id in children_ids_keys]
                stats_lst_ids_lst = [
                    parents_ids_lst[c_id][layer.layer_id] for c_id in children_ids_keys
                ]
                intensity_arr_virtual_ll.append(
                    layer.virtual_ll(
                        None,
                        layer.virtual_intensity_arr,
                        lower_layers_lst,
                        stats_lst_ids_lst,
                    )
                )

        if (
            np.abs(np_sum(intensity_mat_virtual_ll) - np_sum(intensity_arr_virtual_ll))
            > 0.1
            or np.abs(np_sum(intensity_arr_virtual_ll) - sum_virtual_ll) > 0.1
        ):
            print(
                "ex_id = ",
                self.ex_id,
                ", intensity mat virtual ll = ",
                sum(intensity_mat_virtual_ll),
                ", intensity arr virtual ll = ",
                sum(intensity_arr_virtual_ll),
                ", virtual ll = ",
                sum_virtual_ll,
            )

    def fast_virtualMCMCAccept(self, l, rng, swap, parents_ids_lst, children_ids_lst):
        dpp_layers_events = self.dpp_layers_events
        layers = dpp_layers_events.layers
        this_layer = layers[l]
        real_events_num = len(this_layer.real_events)
        virtual_events_num = len(this_layer.virtual_events)
        events_tot_num = real_events_num + virtual_events_num
        if events_tot_num > 0:
            if not swap:
                chosen_index = rng.integers(low=0, high=events_tot_num)
                if chosen_index < real_events_num:
                    (
                        ll_diff_dq,
                        virtual_ll_diff_dq,
                        stats_diff_dq,
                        virtual_stats_diff_dq,
                    ) = neighbor_layers_stats_diff_after_moving_event(
                        dpp_layers_events=dpp_layers_events,
                        this_layer_id=l,
                        chosen_index=chosen_index,
                        event_to_append=None,
                        move_type="del",
                        real_id=None,
                        virtual_id=None,
                    )

                    virtual_event_to_append = this_layer.real_events[chosen_index]
                    children_ids_keys = children_ids_lst[l].keys()
                    lower_layers_lst = [layers[c_id] for c_id in children_ids_keys]

                    stats_lst_ids_lst = [
                        parents_ids_lst[c_id][l] for c_id in children_ids_keys
                    ]

                    virtual_intensity_mat_lst_to_append = calc_virtual_intensity_mat_lst(
                        lower_layers_lst=lower_layers_lst,
                        events_times=virtual_event_to_append,
                        stats_lst_ids_lst=stats_lst_ids_lst,
                    )
                    virtual_intensity_arr_to_append = calc_virtual_intensity_arr(
                        virtual_intensity_mat_lst=virtual_intensity_mat_lst_to_append,
                        virtual_events_times=virtual_event_to_append,
                        virtual_base_rate=this_layer.virtual_base_rate,
                        lower_layers_lst=lower_layers_lst,
                        stats_lst_ids_lst=stats_lst_ids_lst,
                    )
                    virtual_intensity_for_event_to_append = virtual_intensity_arr_to_append[
                        0
                    ]
                    log_virtual_intensity_for_event_to_append = np_log(
                        virtual_intensity_for_event_to_append
                    )
                    if "virtual_kernels_param_lst" in this_layer.__dict__:
                        log_virtual_real_ratio = (
                            log_virtual_intensity_for_event_to_append
                            - np_log(this_layer.intensity_arr[chosen_index])
                        )
                    else:
                        log_virtual_real_ratio = (
                            log_virtual_intensity_for_event_to_append
                            - np_log(this_layer.base_rate)
                        )

                    accept_prob = (
                        sum(ll_diff_dq)
                        + sum(virtual_ll_diff_dq)
                        + log_virtual_real_ratio
                    )

                    mu = rng.uniform()
                    self.jump_attempts += 1
                    if np.log(mu) < accept_prob:
                        self.jump_acceptance += 1
                        change_real_event_neighbor_layers(
                            change_type="del",
                            chosen_index=chosen_index,
                            dpp_layers_events=dpp_layers_events,
                            this_layer_id=l,
                            to_append_event=None,
                            ll_diff_dq=ll_diff_dq,
                            stats_diff_dq=stats_diff_dq,
                            virtual_ll_diff_dq=virtual_ll_diff_dq,
                            virtual_stats_diff_dq=virtual_stats_diff_dq,
                            swap_real_id=None,
                            swap_virtual_id=None,
                            parents_ids_lst=parents_ids_lst,
                            children_ids_lst=children_ids_lst,
                        )
                        flip_real_event_this_layer(
                            this_layer=this_layer,
                            chosen_index=chosen_index,
                            virtual_intensity_mat_lst_to_append=virtual_intensity_mat_lst_to_append,
                            virtual_intensity_arr_to_append=virtual_intensity_for_event_to_append,
                            log_virtual_intensity_arr_to_append=log_virtual_intensity_for_event_to_append,
                        )
                else:
                    chosen_index -= real_events_num
                    (
                        ll_diff_dq,
                        virtual_ll_diff_dq,
                        stats_diff_dq,
                        virtual_stats_diff_dq,
                    ) = neighbor_layers_stats_diff_after_moving_event(
                        dpp_layers_events=dpp_layers_events,
                        this_layer_id=l,
                        chosen_index=chosen_index,
                        event_to_append=this_layer.virtual_events[chosen_index],
                        move_type="append",
                        real_id=None,
                        virtual_id=None,
                    )

                    real_event_to_append = this_layer.virtual_events[chosen_index]
                    parents_ids_keys = parents_ids_lst[l].keys()
                    upper_layers_lst = [layers[p_id] for p_id in parents_ids_keys]

                    stats_lst_ids_lst = [
                        children_ids_lst[u_id][l] for u_id in parents_ids_keys
                    ]

                    intensity_mat_lst_to_append = calc_intensity_mat_lst(
                        upper_layers_lst=upper_layers_lst,
                        events_times=np_array([real_event_to_append]),
                        stats_lst_ids_lst=stats_lst_ids_lst,
                    )
                    if upper_layers_lst:
                        p_min_lst = [
                            np_min(p_layer.real_events)
                            if len(p_layer.real_events) != 0
                            else np_inf
                            for p_layer in upper_layers_lst
                        ]
                        p_min = np_min(p_min_lst)
                    elif "virtual_kernels_param_lst" not in this_layer.__dict__:
                        p_min = 0
                    intensity_arr = calc_intensity_arr(
                        p_min=p_min,
                        intensity_mat_lst=intensity_mat_lst_to_append,
                        events_times=real_event_to_append,
                        base_rate=this_layer.base_rate,
                    )
                    intensity_for_event_to_append = intensity_arr[0]
                    assert np_any(intensity_arr >= 0)
                    if intensity_for_event_to_append == 0:
                        log_real_intensity_to_add = -np_inf
                    else:
                        log_real_intensity_to_add = np_log(
                            intensity_for_event_to_append
                        )
                    log_real_virtual_ratio = log_real_intensity_to_add - np_log(
                        this_layer.virtual_intensity_arr[chosen_index]
                    )

                    accept_prob = (
                        sum(ll_diff_dq)
                        + sum(virtual_ll_diff_dq)
                        + log_real_virtual_ratio
                    )
                    mu = rng.uniform()
                    self.jump_attempts += 1
                    if np.log(mu) < accept_prob:
                        self.jump_acceptance += 1
                        change_real_event_neighbor_layers(
                            change_type="append",
                            chosen_index=None,
                            dpp_layers_events=dpp_layers_events,
                            this_layer_id=l,
                            to_append_event=this_layer.virtual_events[chosen_index],
                            ll_diff_dq=ll_diff_dq,
                            stats_diff_dq=stats_diff_dq,
                            virtual_ll_diff_dq=virtual_ll_diff_dq,
                            virtual_stats_diff_dq=virtual_stats_diff_dq,
                            swap_real_id=None,
                            swap_virtual_id=None,
                            parents_ids_lst=parents_ids_lst,
                            children_ids_lst=children_ids_lst,
                        )
                        flip_virtual_event_this_layer(
                            this_layer=this_layer,
                            chosen_index=chosen_index,
                            intensity_mat_lst_to_append=intensity_mat_lst_to_append,
                            intensity_arr_to_append=intensity_for_event_to_append,
                            log_intensity_arr_to_append=log_real_intensity_to_add,
                        )
            elif swap:
                chosen_real_index = rng.integers(low=0, high=real_events_num)
                chosen_virtual_index = rng.integers(low=0, high=virtual_events_num)
                (
                    ll_diff_dq,
                    virtual_ll_diff_dq,
                    stats_diff_dq,
                    virtual_stats_diff_dq,
                ) = neighbor_layers_stats_diff_after_moving_event(
                    dpp_layers_events=dpp_layers_events,
                    this_layer_id=l,
                    chosen_index=None,
                    event_to_append=None,
                    move_type="swap",
                    real_id=chosen_real_index,
                    virtual_id=chosen_virtual_index,
                )

                virtual_event_to_append = this_layer.real_events[chosen_real_index]
                children_ids_keys = children_ids_lst[l].keys()
                lower_layers_lst = [layers[c_id] for c_id in children_ids_keys]

                stats_lst_ids_lst = [
                    parents_ids_lst[c_id][l] for c_id in children_ids_keys
                ]

                virtual_intensity_mat_lst_to_append = calc_virtual_intensity_mat_lst(
                    lower_layers_lst=lower_layers_lst,
                    events_times=virtual_event_to_append,
                    stats_lst_ids_lst=stats_lst_ids_lst,
                )
                virtual_intensity_arr_to_append = calc_virtual_intensity_arr(
                    virtual_intensity_mat_lst=virtual_intensity_mat_lst_to_append,
                    virtual_events_times=virtual_event_to_append,
                    virtual_base_rate=this_layer.virtual_base_rate,
                    lower_layers_lst=lower_layers_lst,
                    stats_lst_ids_lst=stats_lst_ids_lst,
                )
                real_event_to_append = this_layer.virtual_events[chosen_virtual_index]
                parents_ids_keys = parents_ids_lst[l].keys()
                upper_layers_lst = [layers[p_id] for p_id in parents_ids_keys]

                stats_lst_ids_lst = [
                    children_ids_lst[u_id][l] for u_id in parents_ids_keys
                ]

                intensity_mat_lst_to_append = calc_intensity_mat_lst(
                    upper_layers_lst=upper_layers_lst,
                    events_times=np_array([real_event_to_append]),
                    stats_lst_ids_lst=stats_lst_ids_lst,
                )
                if upper_layers_lst:
                    p_min_lst = [
                        np_min(p_layer.real_events)
                        if len(p_layer.real_events) != 0
                        else np_inf
                        for p_layer in upper_layers_lst
                    ]
                    p_min = np_min(p_min_lst)
                elif "virtual_kernels_param_lst" not in this_layer.__dict__:
                    p_min = 0
                intensity_arr = calc_intensity_arr(
                    p_min=p_min,
                    intensity_mat_lst=intensity_mat_lst_to_append,
                    events_times=real_event_to_append,
                    base_rate=this_layer.base_rate,
                )
                assert np_any(intensity_arr >= 0)
                intensity_for_event_to_append = intensity_arr[0]
                if intensity_arr[0] == 0:
                    log_real_intensity_to_add = -np_inf
                else:
                    log_real_intensity_to_add = np_log(intensity_for_event_to_append)
                virtual_intensity_for_event_to_append = virtual_intensity_arr_to_append[
                    0
                ]
                log_virtual_intensity_to_add = np_log(
                    virtual_intensity_arr_to_append[0]
                )
                if "virtual_kernels_param_lst" in this_layer.__dict__:
                    log_ratio = (
                        log_real_intensity_to_add
                        + log_virtual_intensity_to_add
                        - np_log(this_layer.virtual_intensity_arr[chosen_virtual_index])
                        - np_log(this_layer.intensity_arr[chosen_real_index])
                    )
                else:
                    log_ratio = (
                        log_real_intensity_to_add
                        + log_virtual_intensity_to_add
                        - np_log(this_layer.virtual_intensity_arr[chosen_virtual_index])
                        - np_log(this_layer.base_rate)
                    )

                accept_prob = log_ratio + sum(ll_diff_dq) + sum(virtual_ll_diff_dq)
                mu = rng.uniform()
                self.jump_attempts += 1
                if np.log(mu) < accept_prob:
                    self.jump_acceptance += 1
                    change_real_event_neighbor_layers(
                        change_type="swap",
                        chosen_index=None,
                        dpp_layers_events=dpp_layers_events,
                        this_layer_id=l,
                        to_append_event=this_layer.virtual_events[chosen_virtual_index],
                        ll_diff_dq=ll_diff_dq,
                        stats_diff_dq=stats_diff_dq,
                        virtual_ll_diff_dq=virtual_ll_diff_dq,
                        virtual_stats_diff_dq=virtual_stats_diff_dq,
                        swap_real_id=chosen_real_index,
                        swap_virtual_id=chosen_virtual_index,
                        parents_ids_lst=parents_ids_lst,
                        children_ids_lst=children_ids_lst,
                    )
                    swap_events_this_layer(
                        this_layer,
                        real_id=chosen_real_index,
                        virtual_id=chosen_virtual_index,
                        intensity_arr_to_append=intensity_for_event_to_append,
                        virtual_intensity_arr_to_append=virtual_intensity_for_event_to_append,
                        log_real_intensity_to_add=log_real_intensity_to_add,
                        log_virtual_intensity_to_add=log_virtual_intensity_to_add,
                        intensity_mat_lst_to_append=intensity_mat_lst_to_append,
                        virtual_intensity_mat_lst_to_append=virtual_intensity_mat_lst_to_append,
                    )

    def joint_ll(self):
        ll = [layer.loglikelihood for layer in self.dpp_layers_events.layers]
        return ll

    def joint_virtual_ll(self):
        ll = [
            layer.virtual_loglikelihood
            for layer in self.dpp_layers_events.layers
            if layer.virtual_events is not None
        ]
        return ll

    def mean_of_tot_joint_ll(self):
        return self.tot_joint_ll / self.joint_ll_count

    def mean_of_tot_joint_virtual_ll(self):
        return self.tot_joint_virtual_ll / self.joint_ll_count

    def mean_of_mixed_joint_ll(self):
        return self.mixed_joint_ll / self.mixed_ll_count

    def mean_of_mixed_joint_virtual_ll(self):
        return self.mixed_joint_virtual_ll / self.mixed_ll_count

    def get_jump_attempts(self):
        return self.jump_attempts

    def get_jump_acceptance(self):
        return self.jump_acceptance
