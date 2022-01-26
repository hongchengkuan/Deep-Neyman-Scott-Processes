import numpy as np
from utils import BatchManager
from batchposteriorsampler import BatchPosteriorSampler
from numpy.random import SeedSequence, default_rng
from DPPLayersSamples import DPPLayersSamples

ss = SeedSequence(12345)


class Evaluator:
    def __init__(
        self,
        dpp_layers,
        end_time_tuple,
        num_of_mcmc_samples,
        evidences_dict,
        batch_size,
        dpp_layers_events_factory,
        keep_last_samples,
        time_scale,
        sample_intervals,
        random_sample_layer,
    ):
        self.batchmanager = BatchManager(
            batch_size=batch_size, num_of_evidences=len(evidences_dict), num_of_epochs=1
        )
        child_seeds = ss.spawn(len(evidences_dict))
        rng_list = [default_rng(s) for s in child_seeds]
        self.batchposteriorsampler = BatchPosteriorSampler(
            dpp_layers_events_factory=dpp_layers_events_factory,
            rng_list=rng_list,
            keep_last_samples=keep_last_samples,
            initial_burn_in_steps=0,
            random_sample_layer=random_sample_layer,
        )
        self.dpp_layers_samples = DPPLayersSamples(
            dpp_layers=dpp_layers,
            end_time_tuple=end_time_tuple,
            num_of_mcmc_samples=num_of_mcmc_samples,
            evidences_dict=evidences_dict,
            valid=True,
        )
        self.time_scale = time_scale
        self.tot_evidence_ll = 0
        self.num_of_events_cum_evidence = 0
        self.sample_intervals = sample_intervals
        self.virtual_top_mu_optimizer = AdamOptimizer(alpha=0.1)

    def evaluate(
        self,
        valid_burn_in_steps,
        num_of_events_evidence_each_example,
        log_folder_name=None,
    ):
        evidence_index = []
        dpp_layers = self.dpp_layers_samples.dpp_layers
        for count, l in enumerate(dpp_layers.nontop_ids_lst):
            if not dpp_layers.children_ids_lst[l]:
                evidence_index.append(count)
        virtual_grad_top_wrt_real_accum = 0
        for i in range(self.batchmanager.num_of_iters):
            example_ids = self.batchmanager.example_ids_for_iter(i)

            (
                all_samples_dict,
                all_virtual_samples_dict,
            ) = self.batchposteriorsampler.batch_posterior_samples(
                example_ids,
                burn_in_steps=valid_burn_in_steps,
                e_step_sample_size=self.dpp_layers_samples.num_of_mcmc_samples,
                sample_intervals=self.sample_intervals,
            )
            self.dpp_layers_samples.assign_sample(
                all_samples_dict, all_virtual_samples_dict, example_ids
            )

            # -------------------------------------------------------------------------------
            # debug
            mixed_ll = np.mean(self.batchposteriorsampler.mean_of_mixed_joint_ll)
            ll = np.mean(self.batchposteriorsampler.mean_of_tot_joint_ll)
            mixed_virtual_ll = np.mean(
                self.batchposteriorsampler.mean_of_mixed_joint_virtual_ll
            )
            virtual_ll = np.mean(
                self.batchposteriorsampler.mean_of_tot_joint_virtual_ll
            )
            print(f"mean of ll = {ll}, mean of virtual ll = {virtual_ll}")
            print(
                f"mixed joint ll = {mixed_ll}, mixed virtual joint ll = {mixed_virtual_ll}"
            )

            # self.dpp_layers_samples.check_ll_grad(
            #     example_ids,
            #     jac=True,
            #     virtual_jac=True,
            #     virtual_wrt_real_jac=True,
            #     maximize=True,
            # )
            # -------------------------------------------------------------------------------

            stats = self.dpp_layers_samples.calc_stats_from_samples(
                jac=False, virtual_jac=False, virtual_wrt_real_jac=True, maximize=True,
            )
            virtual_grad_top_wrt_real_accum += stats.virtual_top_grad_mu_wrt_real

            dpp_layers.set_top_mu(
                stats.top_base_rate, example_ids, valid=True, transform=False
            )
            self.tot_evidence_ll += (
                np.sum(stats.ll[evidence_index]) * self.batchmanager.batch_size
            )
            self.num_of_events_cum_evidence += np.sum(
                num_of_events_evidence_each_example[example_ids]
            )
        if dpp_layers.layers[0].virtual_valid_base_rate[0] > 0:
            x0_top_virtual_mu = dpp_layers.get_top_virtual_mu(
                example_ids, valid=True, transform=True
            )
            top_virtual_mu_optimized = self.virtual_top_mu_optimizer.ascent(
                params=x0_top_virtual_mu, g=virtual_grad_top_wrt_real_accum,
            )
            dpp_layers.set_top_virtual_mu(
                params=top_virtual_mu_optimized,
                example_ids=example_ids,
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
            print(np.mean(stats.top_base_rate), file=valid_base_rate_log, flush=True)
            for layer_id in dpp_layers.top_ids_lst:
                print(
                    dpp_layers.layers[layer_id].valid_base_rate.tolist(),
                    file=valid_base_rate_log,
                    flush=True,
                )
        with open(virtual_valid_base_rate_filename, "a") as virtual_valid_base_rate_log:
            if dpp_layers.layers[0].virtual_base_rate[0] > 0:
                print(
                    np.mean(top_virtual_mu_optimized),
                    file=virtual_valid_base_rate_log,
                    flush=True,
                )
            for layer_id in dpp_layers.top_ids_lst:
                print(
                    dpp_layers.layers[
                        layer_id
                    ].virtual_valid_base_rate.tolist(),
                    file=virtual_valid_base_rate_log,
                    flush=True,
                )
        # ------------------------------------------------------------------------------

        ll_per_event = self.tot_evidence_ll / self.num_of_events_cum_evidence
        ll_per_event += np.log(self.time_scale)
        self.tot_evidence_ll = 0
        self.num_of_events_cum_evidence = 0
        return ll_per_event

from inference import AdamOptimizer