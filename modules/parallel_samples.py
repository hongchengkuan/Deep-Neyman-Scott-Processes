from multiprocessing import Pool
from functools import reduce

posterior_sampler_dict = {}
burn_in_steps = 0
e_step_sample_size = 0
sample_intervals = 1
initial_burn_in_steps = 0
predict = False


def get_samples(example_id):

    # assign evidence events
    sampler = posterior_sampler_dict[example_id]
    if sampler.prior_sample is None and initial_burn_in_steps > 0:
        sampler.virtual_mixing(burn_in_steps=initial_burn_in_steps)
    else:
        sampler.virtual_mixing(burn_in_steps)
    samples_dict = {}
    virtual_samples_dict = {}
    if not predict:
        (
            samples_dict_ex,
            virtual_samples_dict_ex,
        ) = sampler.MCMCSampler.virtualMCMCLayers(
            layers_ids_lst=sampler.MCMCLayers_ids_lst,
            virtual_resample_prob=sampler.virtual_resample_prob,
            swap_prob=sampler.swap_prob,
            rng=sampler.rng,
            sample_intervals=sample_intervals,
            num_of_samples=e_step_sample_size,
            burn_in=False,
            random_sample_layer=sampler.random_sample_layer,
            predict=predict,
        )
    elif predict:
        (
            samples_dict_ex,
            virtual_samples_dict_ex,
            first_event_dict_dq_ex,
        ) = sampler.MCMCSampler.virtualMCMCLayers(
            layers_ids_lst=sampler.MCMCLayers_ids_lst,
            virtual_resample_prob=sampler.virtual_resample_prob,
            swap_prob=sampler.swap_prob,
            rng=sampler.rng,
            sample_intervals=sample_intervals,
            num_of_samples=e_step_sample_size,
            burn_in=False,
            random_sample_layer=sampler.random_sample_layer,
            predict=predict,
        )
    samples_dict[example_id] = samples_dict_ex  # layer_id:[sample_0, sample_1, ...]
    virtual_samples_dict[example_id] = virtual_samples_dict_ex

    mean_of_tot_joint_ll = sampler.MCMCSampler.mean_of_tot_joint_ll()
    mean_of_mixed_joint_ll = sampler.MCMCSampler.mean_of_mixed_joint_ll()
    mean_of_tot_joint_virtual_ll = sampler.MCMCSampler.mean_of_tot_joint_virtual_ll()
    mean_of_mixed_joint_virtual_ll = (
        sampler.MCMCSampler.mean_of_mixed_joint_virtual_ll()
    )
    jump_attempts = sampler.MCMCSampler.get_jump_attempts()
    jump_acceptance = sampler.MCMCSampler.get_jump_acceptance()
    if predict:
        first_event_dict_dq_dict = {example_id: first_event_dict_dq_ex}
        return (
            samples_dict,
            virtual_samples_dict,
            mean_of_tot_joint_ll,
            mean_of_mixed_joint_ll,
            mean_of_tot_joint_virtual_ll,
            mean_of_mixed_joint_virtual_ll,
            sampler.rng,
            jump_attempts,
            jump_acceptance,
            first_event_dict_dq_dict,
        )
    else:
        return (
            samples_dict,
            virtual_samples_dict,
            mean_of_tot_joint_ll,
            mean_of_mixed_joint_ll,
            mean_of_tot_joint_virtual_ll,
            mean_of_mixed_joint_virtual_ll,
            sampler.rng,
            jump_attempts,
            jump_acceptance,
        )

def get_all_samples(example_ids, parallel):
    if parallel:
        pool = Pool()
        all_samples = pool.map(get_samples, example_ids)
        pool.close()
        pool.join()
        pool = None
    else:
        all_samples = list(map(get_samples, example_ids))
    all_samples_dict = [s[0] for s in all_samples]
    all_virtual_samples_dict = [s[1] for s in all_samples]
    all_mean_of_tot_joint_ll = [s[2] for s in all_samples]
    all_mean_of_mixed_joint_ll = [s[3] for s in all_samples]
    all_mean_of_tot_joint_virtual_ll = [s[4] for s in all_samples]
    all_mean_of_mixed_joint_virtual_ll = [s[5] for s in all_samples]
    rng_list = [s[6] for s in all_samples]
    jump_attempts = [s[7] for s in all_samples]
    jump_acceptance = [s[8] for s in all_samples]
    all_samples_dict = reduce(lambda a, b: {**a, **b}, all_samples_dict)
    all_virtual_samples_dict = reduce(lambda a, b: {**a, **b}, all_virtual_samples_dict)
    if predict:
        first_event_dict_dq_dict = [s[9] for s in all_samples] # [ex_id * sample_id * {type:time}]
        first_event_dict_dq_dict = reduce(lambda a, b: {**a, **b}, first_event_dict_dq_dict)
        return (
            all_samples_dict,
            all_virtual_samples_dict,
            all_mean_of_tot_joint_ll,
            all_mean_of_mixed_joint_ll,
            all_mean_of_tot_joint_virtual_ll,
            all_mean_of_mixed_joint_virtual_ll,
            rng_list,
            jump_attempts,
            jump_acceptance,
            first_event_dict_dq_dict
        )
    return (
        all_samples_dict,
        all_virtual_samples_dict,
        all_mean_of_tot_joint_ll,
        all_mean_of_mixed_joint_ll,
        all_mean_of_tot_joint_virtual_ll,
        all_mean_of_mixed_joint_virtual_ll,
        rng_list,
        jump_attempts,
        jump_acceptance,
    )
