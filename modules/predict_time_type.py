import numpy as np
from utils import BatchManager
from batchposteriorsampler import BatchPosteriorSampler
from numpy.random import SeedSequence, default_rng
from DPPLayersSamples import DPPLayersSamples

# import DPPLayers
from collections import deque
from copy import deepcopy
from DPPLayers import DPPLayersEvents, sp
from models_factory import dpp_layers_factory
from inference import AdamOptimizer

ss = SeedSequence(12345)
np_array = np.array
np_sort = np.sort
np_max = np.max
np_min = np.min
np_abs = np.abs
np_mean = np.mean
np_append = np.append
np_any = np.any


def predict_time_type(
    model_name,
    evidence_dict,
    num_of_types,
    num_of_layers,
    kernel_type,
    base_rate_init,
    valid_base_rate_init,
    virtual_base_rate_init,
    kernel_params_dq,
    virtual_kernel_params_dq,
    var_ids,
    virtual_var_ids,
    initial_burn_in_steps,
    burn_in_steps,
    num_of_mcmc_samples,
    keep_last_samples,
    time_scale,
    sample_intervals,
    random_sample_layer,
    log_folder_name,
    em_iters,
):
    """ predict time and type for one data point """

    time_pred_square_err_sum = 0
    type_pred_correct_num = 0

    evidences_dict = {0: evidence_dict}
    dpp_layers = dpp_layers_factory(
        model_name=model_name,
        evidences={},
        valid_evidences=evidences_dict,
        num_of_types=num_of_types,
        num_of_layers=num_of_layers,
        kernel_type=kernel_type,
        kernel_params_dq=kernel_params_dq,
        virtual_kernel_params_dq=virtual_kernel_params_dq,
        var_ids=var_ids,
        virtual_var_ids=virtual_var_ids,
        base_rate_init=base_rate_init,
        valid_base_rate_init=valid_base_rate_init,
        virtual_base_rate_init=virtual_base_rate_init,
    )

    evidence_index = []
    for count, l in enumerate(dpp_layers.nontop_ids_lst):
        if not dpp_layers.children_ids_lst[l]:
            evidence_index.append(count)

    seed = ss.spawn(1)
    rng_lst = [default_rng(seed[0])]
    batchposteriorsampler = BatchPosteriorSampler(
        dpp_layers_events_factory=None,
        rng_list=rng_lst,
        keep_last_samples=keep_last_samples,
        initial_burn_in_steps=initial_burn_in_steps,
        random_sample_layer=random_sample_layer,
    )

    evidence_dict_to_predict = deepcopy(evidence_dict)
    evidence_dict_to_predict = {
        k: np_sort(v) for k, v in evidence_dict_to_predict.items()
    }
    num_of_events = sum([len(evidence_dict[i]) for i in evidence_dict])
    num_of_events_to_predict = num_of_events - 1
    known_evidence_dict = {k: np_array([]) for k in evidence_dict}
    potential_next_events = {
        k: v[0] for k, v in evidence_dict_to_predict.items() if len(v) != 0
    }
    potential_next_events = {
        k: v for k, v in sorted(potential_next_events.items(), key=lambda item: item[1])
    }
    next_event_type = next(iter(potential_next_events))
    next_event_time = next(iter(potential_next_events.values()))
    known_evidence_dict[next_event_type] = np_append(
        known_evidence_dict[next_event_type], next_event_time
    )
    evidence_dict_to_predict[next_event_type] = evidence_dict_to_predict[
        next_event_type
    ][1:]
    last_time = next_event_time

    em_iters_count = em_iters
    predict_count = 0

    while num_of_events_to_predict >= 0:
        dpp_layers_samples = DPPLayersSamples(
            dpp_layers=dpp_layers,
            end_time_tuple=(last_time,),
            num_of_mcmc_samples=num_of_mcmc_samples,
            evidences_dict={0: known_evidence_dict},
            valid=True,
        )

        virtual_top_mu_optimizer = AdamOptimizer(alpha=0.1)
        while True:
            dpp_layers_events = DPPLayersEvents(
                dpp_layers=dpp_layers,
                ex_id=0,
                valid=True,
                end_time=last_time,
                evidences_this_ex=known_evidence_dict,
            )
            if em_iters_count == 0:
                em_iters_count = em_iters

                (
                    all_samples_dict,
                    all_virtual_samples_dict,
                    all_first_event_dic_dq,
                ) = batchposteriorsampler.batch_posterior_samples(
                    example_ids=[0],
                    burn_in_steps=burn_in_steps,
                    e_step_sample_size=num_of_mcmc_samples,
                    sample_intervals=sample_intervals,
                    dpp_layers_events=dpp_layers_events,
                    parallel=False,
                    predict=True,
                    check_parents=True,
                )
                if num_of_events_to_predict != 0:
                    first_event_dic_dq = all_first_event_dic_dq[0]
                    potential_next_events = {
                        k: v[0]
                        for k, v in evidence_dict_to_predict.items()
                        if len(v) != 0
                    }
                    potential_next_events = {
                        k: v
                        for k, v in sorted(
                            potential_next_events.items(), key=lambda item: item[1]
                        )
                    }
                    next_event_type = next(iter(potential_next_events))
                    next_event_time = next(iter(potential_next_events.values()))
                    type_pred_count_dict = {
                        e_layer_id: 0 for e_layer_id in dpp_layers.evidences_ids_set
                    }
                    pred_time_dq = deque([])
                    pred_time_dq_append = pred_time_dq.append

                    for dic in first_event_dic_dq:
                        type_pred_count_dict[dic["type"]] += 1
                        pred_time_dq_append(dic["time"])
                    type_pred_count_dict = {
                        k: v
                        for k, v in sorted(
                            type_pred_count_dict.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    }
                    pred_next_type = next(iter(type_pred_count_dict))
                    assert not np_any(last_time > pred_time_dq)
                    if pred_next_type == next_event_type:
                        type_pred_correct_num += 1
                    time_pred_square_err_sum += (
                        next_event_time - np_mean(pred_time_dq)
                    ) ** 2
                    predict_count += 1

                    known_evidence_dict[next_event_type] = np_append(
                        known_evidence_dict[next_event_type], next_event_time
                    )
                    evidence_dict_to_predict[
                        next_event_type
                    ] = evidence_dict_to_predict[next_event_type][1:]
                    last_time = next_event_time

                elif num_of_events_to_predict == 0:
                    stats = dpp_layers_samples.calc_stats_from_samples(
                        jac=False,
                        virtual_jac=False,
                        virtual_wrt_real_jac=False,
                        maximize=False,
                    )

                    tot_evidence_ll = np.sum(stats.ll[evidence_index])
                    ex_prediction_name = log_folder_name + "prediction.log"
                    with open(ex_prediction_name, "w") as f:
                        print(type_pred_correct_num, file=f, flush=True)
                        print(time_pred_square_err_sum, file=f, flush=True)
                        print(tot_evidence_ll, file=f, flush=True)
                        print(time_scale, file=f, flush=True)
                        print(num_of_events, file=f, flush=True)
                        print(predict_count, file=f, flush=True)

                num_of_events_to_predict -= 1
                break
            (
                all_samples_dict,
                all_virtual_samples_dict,
            ) = batchposteriorsampler.batch_posterior_samples(
                example_ids=[0],
                burn_in_steps=burn_in_steps,
                e_step_sample_size=num_of_mcmc_samples,
                sample_intervals=sample_intervals,
                dpp_layers_events=dpp_layers_events,
                parallel=False,
                predict=False,
                check_parents=True,
            )
            dpp_layers_samples.assign_sample(
                layers_samples=all_samples_dict,
                layers_virtual_samples=all_virtual_samples_dict,
                example_ids=[0],
            )
            em_iters_count -= 1

            # -------------------------------------------------------------------------------
            # debug
            mixed_ll = np_mean(batchposteriorsampler.mean_of_mixed_joint_ll)

            ll = np_mean(batchposteriorsampler.mean_of_tot_joint_ll)
            mixed_virtual_ll = np_mean(
                batchposteriorsampler.mean_of_mixed_joint_virtual_ll
            )
            virtual_ll = np_mean(batchposteriorsampler.mean_of_tot_joint_virtual_ll)
            print(f"mean of ll = {ll}, mean of virtual ll = {virtual_ll}")
            print(
                f"mixed joint ll = {mixed_ll}, mixed virtual joint ll = {mixed_virtual_ll}"
            )
            print(f"num of events to predict = {num_of_events_to_predict}")

            # dpp_layers_samples.check_ll_grad(
            #     example_ids=[0],
            #     jac=True,
            #     virtual_jac=True,
            #     virtual_wrt_real_jac=True,
            #     maximize=True,
            # )
            # -------------------------------------------------------------------------------

            stats = dpp_layers_samples.calc_stats_from_samples(
                jac=False,
                virtual_jac=False,
                virtual_wrt_real_jac=True,
                maximize=True,
                parallel=False,
            )
            virtual_grad_top_wrt_real_accum = stats.virtual_top_grad_mu_wrt_real

            dpp_layers.set_top_mu(
                stats.top_base_rate, example_ids=[0], valid=True, transform=False
            )

            if dpp_layers.layers[0].virtual_valid_base_rate[0] > 0:
                x0_top_virtual_mu = dpp_layers.get_top_virtual_mu(
                    example_ids=[0], valid=True, transform=True
                )
                top_virtual_mu_optimized = virtual_top_mu_optimizer.ascent(
                    params=x0_top_virtual_mu, g=virtual_grad_top_wrt_real_accum,
                )
                dpp_layers.set_top_virtual_mu(
                    params=top_virtual_mu_optimized,
                    example_ids=[0],
                    valid=True,
                    transform=True,
                )

            # debug
            # ------------------------------------------------------------------------------
            valid_base_rate_filename = log_folder_name + "valid_base_rate.log"
            virtual_valid_base_rate_filename = (
                log_folder_name + "virtual_valid_base_rate.log"
            )
            with open(valid_base_rate_filename, "a") as valid_base_rate_log:
                print(
                    np.mean(stats.top_base_rate), file=valid_base_rate_log, flush=True
                )
                for layer_id in dpp_layers.top_ids_lst:
                    print(
                        dpp_layers.layers[layer_id].valid_base_rate.tolist(),
                        file=valid_base_rate_log,
                        flush=True,
                    )
            with open(
                virtual_valid_base_rate_filename, "a"
            ) as virtual_valid_base_rate_log:
                if dpp_layers.layers[0].virtual_valid_base_rate[0] > 0:
                    print(
                        np_mean(sp.fun(top_virtual_mu_optimized)),
                        file=virtual_valid_base_rate_log,
                        flush=True,
                    )
                for layer_id in dpp_layers.top_ids_lst:
                    print(
                        dpp_layers.layers[layer_id].virtual_valid_base_rate.tolist(),
                        file=virtual_valid_base_rate_log,
                        flush=True,
                    )
            # ------------------------------------------------------------------------------

