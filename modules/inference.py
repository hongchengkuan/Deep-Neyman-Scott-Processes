import DPPLayers
from DPPLayersSamples import DPPLayersSamples
import numpy as np
from numpy.random import SeedSequence, default_rng
import time
import warnings
from batchposteriorsampler import BatchPosteriorSampler
from utils import BatchManager

ss = SeedSequence(12345)

warnings.filterwarnings("error")


class Results:
    def __init__(self):
        return


class AdamOptimizer:
    """
    This is the class for doing Adam optimization.
    """
    def __init__(self, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.m = 0
        self.v = 0
        self.t = 0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.alpha = alpha
        self.epsilon = epsilon

    def ascent(self, params, g):
        self.t += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * g
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * g * g
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)
        params += self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params


from model_evidence_ll_evaluator import Evaluator


class EM:
    """
    This is the class for the MCEM.

    Args:
        dpp_layers: the model used to do the inference
        num_of_epochs: the total number of epochs for the optimization
        sample_size: the number of samples for the MCMC 
        sample_intervals: the number of samples we drop between consecutive samples
        valid_sample_size: the number of samples for the validation
        valid_sample_intervals: the number of samples we drop between consecutive samples for validation
        log_folder_name: the name for the folder to store the log
        initial_burn_in_steps: the burn in steps for the first epoch
        burn_in_steps: burn-in steps after the initial burn-in
        valid_burn_in_steps: burn-in steps after the initial burn-in for validation
        evidences: the observed events
        valid_evidences: the observed events in the validation
        end_time_tuple: a tuple used to store the end time for each sequence
        valid_end_time_tuple: a tuple used to store the end time for each sequence for validation
        batch_size: the number of sequences for each batch
        valid_batch_size: the number of sequences for each batch for validation
        dpp_layers_events_factory: the factory used to produce the model
        valid_dpp_layers_events_factory: the factory used to produce the model for validation
        keep_last_samples: whether to keep the last sample from the last Markov chain as the intial state for the next Markov chain 
        opt: whether to do optimization
        num_of_events_evidence_each_example: the number of events for each sequence
        valid_num_of_events_evidence_each_example: the number of events for each sequence for validation
        time_scale: the scale of the time of the events 
        alpha: alpha for adam
        virtual_alpha: alpha for adam for virtual point processes
        optimization_method: "adam",
        track_period: trach the statistics each track_period
        track_validation: whether to track the statistics for validation
        random_sample_layer: whether to sample the layers in a random order
        fix_kernel_params: whether to fix the kernel parameters

    """
    def __init__(
        self,
        dpp_layers,
        num_of_epochs,
        sample_size,
        sample_intervals,
        valid_sample_size,
        valid_sample_intervals,
        log_folder_name,
        initial_burn_in_steps,
        burn_in_steps,
        valid_burn_in_steps,
        evidences,
        valid_evidences,
        end_time_tuple,
        valid_end_time_tuple,
        batch_size,
        valid_batch_size,
        dpp_layers_events_factory,
        valid_dpp_layers_events_factory,
        keep_last_samples=False,
        opt=False,
        num_of_events_evidence_each_example=None,
        valid_num_of_events_evidence_each_example=None,
        time_scale=None,
        alpha=0.001,
        virtual_alpha=0.001,
        optimization_method="adam",
        track_period=1,
        track_validation=True,
        random_sample_layer=True,
        fix_kernel_params=False,
    ):
        self.burn_in_steps = burn_in_steps
        self.e_step_sample_size = sample_size
        self.sample_intervals = sample_intervals
        self.last_samples_dict = {}
        self.last_virtual_samples_dict = {}
        self.log_folder_name = log_folder_name
        self.opt = opt
        if not self.opt:
            self.parameters_arr = None
        self.prior_sampler_list = None
        child_seeds = ss.spawn(len(evidences))
        rng_list = [default_rng(s) for s in child_seeds]
        self.optimizer = AdamOptimizer(alpha=alpha)
        self.virtual_kernel_optimizer = AdamOptimizer(alpha=virtual_alpha)
        self.virtual_top_mu_optimizer = AdamOptimizer(alpha=virtual_alpha)
        self.num_of_events_evidence_each_example = np.array(
            num_of_events_evidence_each_example
        )
        self.valid_num_of_events_evidence_each_example = np.array(
            valid_num_of_events_evidence_each_example
        )
        self.num_of_events_evidence = np.sum(num_of_events_evidence_each_example)
        self.time_scale = time_scale
        self.dpp_layers_samples = DPPLayersSamples(
            dpp_layers=dpp_layers,
            end_time_tuple=end_time_tuple,
            num_of_mcmc_samples=sample_size,
            evidences_dict=evidences,
            valid=False,
        )
        self.batch_manager = BatchManager(
            batch_size=batch_size,
            num_of_evidences=len(evidences),
            num_of_epochs=num_of_epochs,
        )
        self.batchposteriorsampler = BatchPosteriorSampler(
            dpp_layers_events_factory=dpp_layers_events_factory,
            rng_list=rng_list,
            keep_last_samples=keep_last_samples,
            initial_burn_in_steps=initial_burn_in_steps,
            random_sample_layer=random_sample_layer,
        )
        if track_validation:
            self.valid_evidences_ll_evaluator = Evaluator(
                dpp_layers=dpp_layers,
                end_time_tuple=valid_end_time_tuple,
                num_of_mcmc_samples=valid_sample_size,
                evidences_dict=valid_evidences,
                batch_size=valid_batch_size,
                dpp_layers_events_factory=valid_dpp_layers_events_factory,
                keep_last_samples=keep_last_samples,
                time_scale=time_scale,
                sample_intervals=valid_sample_intervals,
                random_sample_layer=random_sample_layer,
            )
        self.valid_burn_in_steps = valid_burn_in_steps
        self.optimization_method = optimization_method
        self.tot_evidence_ll = 0
        self.num_of_events_cum_evidence = 0
        self.time_iterations_init = 0
        self.track_period = track_period
        self.track_validation = track_validation
        self.fix_kernel_params = fix_kernel_params

    def expectation_step(self, example_ids, burn_in_steps):
        (
            all_samples_dict,
            all_virtual_samples_dict,
        ) = self.batchposteriorsampler.batch_posterior_samples(
            example_ids,
            burn_in_steps=burn_in_steps,
            e_step_sample_size=self.e_step_sample_size,
            sample_intervals=self.sample_intervals,
        )
        self.dpp_layers_samples.assign_sample(
            all_samples_dict, all_virtual_samples_dict, example_ids
        )

    def print_settings(self):
        settings_log_filename = self.log_folder_name + "settings.log"
        with open(settings_log_filename, "w") as settings_log:
            print(
                f"num_of_epochs = {self.batch_manager.num_of_epochs}\n"
                f"sample size = {self.e_step_sample_size}\n"
                f"burn in steps = {self.burn_in_steps}\n"
                f"opt = {self.opt}\n",
                file=settings_log,
                flush=True,
            )

    def print_mean_of_tot_joint_ll(self):
        mixed_ll = np.mean(self.batchposteriorsampler.mean_of_mixed_joint_ll)
        ll = np.mean(self.batchposteriorsampler.mean_of_tot_joint_ll)
        mixed_virtual_ll = np.mean(
            self.batchposteriorsampler.mean_of_mixed_joint_virtual_ll
        )
        virtual_ll = np.mean(self.batchposteriorsampler.mean_of_tot_joint_virtual_ll)
        mean_of_joint_ll_filename = self.log_folder_name + "joint_ll.log"
        mean_of_mixed_ll_filename = self.log_folder_name + "mixed_ll.log"
        with open(mean_of_joint_ll_filename, "a") as mean_of_joint_ll_log:
            print(
                "ll = ",
                ll,
                ", virtual ll = ",
                virtual_ll,
                file=mean_of_joint_ll_log,
                flush=True,
            )
        with open(mean_of_mixed_ll_filename, "a") as mean_of_mixed_ll_log:
            print(
                "mixed ll = ",
                mixed_ll,
                ", mixed virtual ll = ",
                mixed_virtual_ll,
                file=mean_of_mixed_ll_log,
                flush=True,
            )
        print(f"mean of ll = {ll}, mean of virtual ll = {virtual_ll}")
        print(
            f"mixed joint ll = {mixed_ll}, mixed virtual joint ll = {mixed_virtual_ll}"
        )

    def print_acceptance_ratio(self):
        jump_attempts = np.sum(self.batchposteriorsampler.jump_attemps)
        jump_acceptance = np.sum(self.batchposteriorsampler.jump_acceptance)
        acceptance_ratio = jump_acceptance / jump_attempts if jump_attempts != 0 else 0
        filename = self.log_folder_name + "acceptance.log"
        with open(filename, "a") as acceptance_log:
            print(acceptance_ratio, file=acceptance_log, flush=True)

    def print_evidence_ll(self, ll_each_nontop_layer, example_ids):
        evidence_index = []
        for count, l in enumerate(self.dpp_layers_samples.dpp_layers.nontop_ids_lst):
            if not self.dpp_layers_samples.dpp_layers.children_ids_lst[l]:
                evidence_index.append(count)
        self.tot_evidence_ll = (
            np.sum(ll_each_nontop_layer[evidence_index]) * self.batch_manager.batch_size
        )
        ll_per_event = self.tot_evidence_ll / np.sum(self.num_of_events_evidence_each_example[example_ids])
        ll_per_event += np.log(self.time_scale)
        filename = self.log_folder_name + "evidence_per_event_ll.log"
        with open(filename, "a") as evidence_ll_log:
            print(ll_per_event, file=evidence_ll_log)
            print(
                time.time() - self.time_iterations_init,
                file=evidence_ll_log,
                flush=True,
            )

    def virtual_optimize(
        self,
        virtual_grad_nontop_wrt_real,
        virtual_func_nontop_wrt_real_val,
        virtual_grad_top_wrt_real,
        virtual_func_top_wrt_real_val,
        example_ids,
        virtual_base_rate_log,
        verbose=True,
        opt_kernel_params=False,
    ):
        x0_virtual_kernel_params = self.dpp_layers_samples.dpp_layers.get_virtual_params(
            transform=True
        )

        results = Results()
        if opt_kernel_params and not self.fix_kernel_params:
            results.x = self.virtual_kernel_optimizer.ascent(
                params=x0_virtual_kernel_params, g=virtual_grad_nontop_wrt_real
            )
            self.dpp_layers_samples.dpp_layers.set_virtual_params(results.x, transform=True)
        else:
            results.x = x0_virtual_kernel_params
        if self.dpp_layers_samples.dpp_layers.layers[0].virtual_base_rate[0] > 0:
            x0_top_virtual_mu = self.dpp_layers_samples.dpp_layers.get_top_virtual_mu(
                example_ids, valid=False, transform=True
            )
            top_virtual_mu_optimized = self.virtual_top_mu_optimizer.ascent(
                params=x0_top_virtual_mu, g=virtual_grad_top_wrt_real,
            )
            self.dpp_layers_samples.dpp_layers.set_top_virtual_mu(
                params=top_virtual_mu_optimized,
                example_ids=example_ids,
                valid=False,
                transform=True,
            )
        if verbose:
            iter = 0
            print(
                "opt iter",
                iter,
                ", nontop func value = ",
                np.sum(virtual_func_nontop_wrt_real_val),
                ", top func value = ",
                np.sum(virtual_func_top_wrt_real_val),
                ", grad = ",
                virtual_grad_nontop_wrt_real,
                "x=",
                DPPLayers.sp.fun(results.x),
            )

            print(
                np.mean(
                    [
                        self.dpp_layers_samples.dpp_layers.layers[
                            layer_id
                        ].virtual_base_rate
                        for layer_id in self.dpp_layers_samples.dpp_layers.top_ids_lst
                    ]
                ),
                file=virtual_base_rate_log,
                flush=True,
            )
            for layer_id in self.dpp_layers_samples.dpp_layers.top_ids_lst:
                if (
                    self.dpp_layers_samples.dpp_layers.layers[0].virtual_base_rate[0]
                    > 0
                ):
                    print(
                        np.mean(DPPLayers.sp.fun(top_virtual_mu_optimized)),
                        file=virtual_base_rate_log,
                        flush=True,
                    )
                print(
                    self.dpp_layers_samples.dpp_layers.layers[
                        layer_id
                    ].virtual_base_rate,
                    file=virtual_base_rate_log,
                    flush=True,
                )
            return results

    def optimize(self, grad, func_val, verbose=True):
        x0 = self.dpp_layers_samples.dpp_layers.get_real_kernel_params(transform=True)

        results = Results()
        results.x = self.optimizer.ascent(params=x0, g=grad)
        self.dpp_layers_samples.dpp_layers.set_real_kernel_params(
            results.x, transform=True
        )
        results.fun = np.sum(func_val)
        if verbose:
            iter = 0
            print(
                "opt iter",
                iter,
                ", func value = ",
                results.fun,
                ", grad = ",
                grad,
                "x=",
                DPPLayers.sp.fun(results.x),
            )
        return results

    def iterations_withoutparents(self):
        self.print_settings()
        params_log_filename = self.log_folder_name + "params_track.log"
        params_log = open(params_log_filename, "w")
        virtual_params_log_filename = self.log_folder_name + "virtual_params_track.log"
        virtual_params_log = open(virtual_params_log_filename, "w")
        base_rate_log_filename = self.log_folder_name + "base_rate.log"
        base_rate_log = open(base_rate_log_filename, "w")
        virtual_base_rate_log_filename = self.log_folder_name + "virtual_base_rate.log"
        virtual_base_rate_log = open(virtual_base_rate_log_filename, "w")
        events_number_log_filename = self.log_folder_name + "events_number.log"
        events_number_log = open(events_number_log_filename, "w")
        grad_log_filename = self.log_folder_name + "grad_track.log"
        grad_log = open(grad_log_filename, "w")
        hidden_events_log_filename = self.log_folder_name + "hidden_events.log"
        hidden_events_log = open(hidden_events_log_filename, "w")
        valid_evidence_ll_log_filename = self.log_folder_name + "valid_evidence_ll.log"
        valid_evidence_ll_log = open(valid_evidence_ll_log_filename, "w")

        self.time_iterations_init = time.time()

        burn_in_steps = self.burn_in_steps
        if self.batch_manager.batch_size >= len(self.dpp_layers_samples.evidences_dict):
            opt_kernel_params = False
            ll_cum_prev = 1e-10
        else:
            opt_kernel_params = True
        for i in range(self.batch_manager.num_of_epochs):
            print(f"{i}th epoch")

            self.batch_manager.shuffle()
            ll_cum = 0
            ll_top_cum = 0

            grad_accum = 0
            func_val_accum = 0

            virtual_grad_nontop_wrt_real_accum = 0
            virtual_func_nontop_wrt_real_val_accum = 0
            virtual_grad_top_wrt_real_accum = 0
            virtual_func_top_wrt_real_val_accum = 0

            for j in range(self.batch_manager.num_of_iters):
                print(f"{j}th iteration")

                example_ids = self.batch_manager.example_ids_for_iter(j)

                print("batch example size = ", len(example_ids))
                self.expectation_step(example_ids, burn_in_steps)

                # bebug purpose -----------------------------------------
                self.print_mean_of_tot_joint_ll()
                self.print_acceptance_ratio()

                # time_init_func_val = time.time()
                # self.dpp_layers_samples.check_ll_grad(
                #     example_ids=example_ids,
                #     jac=True,
                #     virtual_jac=True,
                #     virtual_wrt_real_jac=True,
                #     maximize=True,
                # )
                # print(f"init func value time = {time.time() - time_init_func_val}")

                # debug -------------------------------------------------

                # update base rate for top layer ------------------------
                stats = self.dpp_layers_samples.calc_stats_from_samples(
                    jac=True,
                    virtual_jac=False,
                    virtual_wrt_real_jac=True,
                    maximize=True,
                )
                self.dpp_layers_samples.dpp_layers.set_top_mu(
                    stats.top_base_rate, example_ids, valid=False, transform=False
                )
                ll_top_cum += np.sum(np.mean(stats.top_ll, axis=0))
                print(np.mean(stats.top_base_rate), file=base_rate_log, flush=True)
                for layer_id in self.dpp_layers_samples.dpp_layers.top_ids_lst:
                    print(
                        self.dpp_layers_samples.dpp_layers.layers[layer_id].base_rate,
                        file=base_rate_log,
                        flush=True,
                    )

                # -------------------------------------------------------
                # debug top layer -------------------------------------
                # stats_top_debug = self.dpp_layers_samples.calc_stats_from_samples(
                #     jac=True,
                #     virtual_jac=False,
                #     virtual_wrt_real_jac=False,
                #     maximize=True,
                # )
                # print(
                #     f"init ll top = {np.sum(np.mean(stats.top_ll, axis=0))}, init grad = {stats.top_mu_grad}, "
                #     f"max ll top = {np.sum(np.mean(stats_top_debug.top_ll, axis=0))}, max grad = {stats_top_debug.top_mu_grad}"
                # )

                # -----------------------------------------------------------------
                func_val_accum = stats.ll
                grad_accum = stats.grad

                virtual_grad_nontop_wrt_real_accum = stats.virtual_grad_wrt_real
                virtual_func_nontop_wrt_real_val_accum = stats.virtual_ll_wrt_real
                virtual_grad_top_wrt_real_accum = stats.virtual_top_grad_mu_wrt_real
                virtual_func_top_wrt_real_val_accum = stats.virtual_top_ll_wrt_real

                if self.batch_manager.batch_size >= len(self.dpp_layers_samples.evidences_dict):
                    ll_cum = np.sum(func_val_accum) + ll_top_cum
                    if not opt_kernel_params:
                        if np.abs((ll_cum_prev - ll_cum) / ll_cum_prev) < 1e-2:
                            opt_kernel_params = True
                    ll_cum_prev = ll_cum

                if self.optimization_method == "adam":
                    print(
                        "real grad = ",
                        grad_accum,
                        "virtual wrt real grad = ",
                        virtual_grad_nontop_wrt_real_accum,
                        file=grad_log,
                        flush=True,
                    )
                    if opt_kernel_params and not self.fix_kernel_params:
                        results = self.optimize(grad=grad_accum, func_val=func_val_accum)
                    virtual_results = self.virtual_optimize(
                        virtual_grad_nontop_wrt_real=virtual_grad_nontop_wrt_real_accum,
                        virtual_func_nontop_wrt_real_val=virtual_func_nontop_wrt_real_val_accum,
                        virtual_grad_top_wrt_real=virtual_grad_top_wrt_real_accum,
                        virtual_func_top_wrt_real_val=virtual_func_top_wrt_real_val_accum,
                        example_ids=example_ids,
                        virtual_base_rate_log=virtual_base_rate_log,
                        opt_kernel_params=opt_kernel_params,
                    )
                else:
                    print("not implemented")
                    return
                # print(
                #     f"init mixed joint ll sum = {init_func_val + np.mean(init_ll_top)}"
                # )
                # print(f"after opt mixed ll sum = {np.mean(max_ll_top) + results.fun}")

                if self.opt:
                    if i % self.track_period == 0:
                        if opt_kernel_params and not self.fix_kernel_params:
                            print(
                                DPPLayers.sp.fun(results.x).tolist(),
                                file=params_log,
                                flush=True,
                            )
                        print(
                            DPPLayers.sp.fun(virtual_results.x).tolist(),
                            file=virtual_params_log,
                            flush=True,
                        )

                self.print_evidence_ll(func_val_accum, example_ids)

                tot_time_till_now = time.time() - self.time_iterations_init
                print(f"total time = {tot_time_till_now}")
                print(tot_time_till_now, file=params_log, flush=True)

                if self.track_validation:
                    if i % self.track_period == 0:
                        valid_ll = self.valid_evidences_ll_evaluator.evaluate(
                            self.valid_burn_in_steps,
                            self.valid_num_of_events_evidence_each_example,
                            log_folder_name=self.log_folder_name,
                        )
                        print(valid_ll, file=valid_evidence_ll_log, flush=True)
                        print(
                            time.time() - self.time_iterations_init,
                            file=valid_evidence_ll_log,
                            flush=True,
                        )
                print("ll non top cum = ", np.sum(func_val_accum))
                print("ll top cum = ", ll_top_cum)
                print("ll cum = ", ll_cum)

            # The following code can be used to print the hidden events -------------------------------------
            # if i % 10 == 0:
            #     sample_index = 0
            #     all_layers_ids = self.dpp_layers_samples.dpp_layers.top_ids_lst + self.dpp_layers_samples.dpp_layers.nontop_ids_lst
            #     hidden_layers_ids = [layer_id for layer_id in all_layers_ids if layer_id not in self.dpp_layers_samples.dpp_layers.evidences_ids_set]
            #     for s in range(self.dpp_layers_samples.num_of_mcmc_samples):
            #         # for l in self.dpp_layers_samples.layers_samples[sample_index]:
            #         for l in hidden_layers_ids:
            #             if len(self.dpp_layers_samples.dpp_layers.children_ids_lst[l])!=0:
            #                 print(
            #                     self.dpp_layers_samples.layers_samples[sample_index][l][s]
            #                     .tolist(),
            #                     file=hidden_events_log,
            #                     flush=True,
            #                 )
            #     # for l in self.dpp_layers_samples.dpp_layers.nontop_ids_lst:
            #     for l in self.dpp_layers_samples.dpp_layers.evidences_ids_set:
            #         if l in self.dpp_layers_samples.evidences_dict[sample_index]:
            #             print(
            #                 self.dpp_layers_samples.evidences_dict[sample_index][l]
            #                 .tolist(),
            #                 file=hidden_events_log,
            #                 flush=True,
            #             )
            #         else:
            #             print(
            #                 [],
            #                 file=hidden_events_log,
            #                 flush=True,
            #             )
            #--------------------------------------------------------------------------------------------------------------------------

        params_log.close()
        events_number_log.close()
        grad_log.close()
        base_rate_log.close()
        hidden_events_log.close()
