import parallel_samples
import numpy as np
import Sampler
import time


class BatchPosteriorSampler:
    """
    This is a class for parallel posterior sampling for a batch of examples.
    """
    def __init__(
        self,
        dpp_layers_events_factory,
        rng_list,
        keep_last_samples,
        initial_burn_in_steps,
        random_sample_layer,
    ):
        self.last_samples_dict = {}
        self.last_virtual_samples_dict = {}
        self.rng_list = rng_list
        self.mean_of_tot_joint_ll = -np.inf
        self.mean_of_mixed_joint_ll = -np.inf
        self.dpp_layers_events_factory = dpp_layers_events_factory
        self.keep_last_samples = keep_last_samples
        self.initial_burn_in_steps = initial_burn_in_steps
        self.random_sample_layer = random_sample_layer

    def batch_posterior_samples(
        self,
        example_ids,
        burn_in_steps,
        e_step_sample_size,
        sample_intervals,
        dpp_layers_events=None,
        parallel=True,
        predict=False,
        check_parents=False,
    ):
        parallel_samples.burn_in_steps = burn_in_steps
        parallel_samples.e_step_sample_size = e_step_sample_size
        parallel_samples.sample_intervals = sample_intervals
        parallel_samples.initial_burn_in_steps = self.initial_burn_in_steps
        parallel_samples.predict = predict
        time1 = time.time()

        def posterior_sampler_factory(ex_id):
            if ex_id in self.last_samples_dict:
                sampler = Sampler.PosteriorSampler(
                    dpp_layers_events=self.dpp_layers_events_factory(ex_id)
                    if dpp_layers_events is None
                    else dpp_layers_events,
                    rng=self.rng_list[ex_id],
                    prior_sample=self.last_samples_dict[ex_id],
                    prior_virtual_sample=self.last_virtual_samples_dict[ex_id],
                    virtual=True,
                    ex_id=ex_id,
                    random_sample_layer=self.random_sample_layer,
                    check_parents=check_parents,
                )
            else:
                sampler = Sampler.PosteriorSampler(
                    dpp_layers_events=self.dpp_layers_events_factory(ex_id)
                    if dpp_layers_events is None
                    else dpp_layers_events,
                    rng=self.rng_list[ex_id],
                    virtual=True,
                    ex_id=ex_id,
                    random_sample_layer=self.random_sample_layer,
                    check_parents=check_parents,
                )
            return sampler

        parallel_samples.posterior_sampler_dict = {
            ex_id: posterior_sampler_factory(ex_id) for ex_id in example_ids
        }
        if not predict:
            (
                all_samples_dict,
                all_virtual_samples_dict,
                all_mean_of_tot_joint_ll,
                all_mean_of_mixed_joint_ll,
                all_mean_of_tot_joint_virtual_ll,
                all_mean_of_mixed_joint_virtual_ll,
                rng_list,
                all_jump_attempts,
                all_jump_acceptance,
            ) = parallel_samples.get_all_samples(example_ids, parallel)
        elif predict:
            (
                all_samples_dict,
                all_virtual_samples_dict,
                all_mean_of_tot_joint_ll,
                all_mean_of_mixed_joint_ll,
                all_mean_of_tot_joint_virtual_ll,
                all_mean_of_mixed_joint_virtual_ll,
                rng_list,
                all_jump_attempts,
                all_jump_acceptance,
                all_first_event_dict_dq
            ) = parallel_samples.get_all_samples(example_ids, parallel)

        for count, ex_id in enumerate(example_ids):
            self.rng_list[ex_id] = rng_list[count]
        self.mean_of_tot_joint_ll = all_mean_of_tot_joint_ll
        self.mean_of_mixed_joint_ll = all_mean_of_mixed_joint_ll
        self.mean_of_tot_joint_virtual_ll = all_mean_of_tot_joint_virtual_ll
        self.mean_of_mixed_joint_virtual_ll = all_mean_of_mixed_joint_virtual_ll
        self.jump_attemps = all_jump_attempts
        self.jump_acceptance = all_jump_acceptance
        print(f"sample paralle time = {time.time() - time1}")
        if self.keep_last_samples:
            for k_1, v_1 in all_samples_dict.items():
                self.last_samples_dict[k_1] = {}
                for k_2, v_2 in v_1.items():
                    self.last_samples_dict[k_1][k_2] = v_2[len(v_2) - 1]
            for k_1, v_1 in all_virtual_samples_dict.items():
                self.last_virtual_samples_dict[k_1] = {}
                for k_2, v_2 in v_1.items():
                    self.last_virtual_samples_dict[k_1][k_2] = v_2[len(v_2) - 1]

        if not predict:
            return all_samples_dict, all_virtual_samples_dict
        elif predict:
            return all_samples_dict, all_virtual_samples_dict, all_first_event_dict_dq
