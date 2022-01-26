import numpy as np
from collections import namedtuple

from parallel_ll_grad_layers import initialize_global_vars, parallel_ll_grad

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

Stats_collections = namedtuple(
    "Stats_collections",
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


class DPPLayersSamples:
    def __init__(
        self,
        dpp_layers,  # this dpp_layers should include all the layers
        end_time_tuple,
        num_of_mcmc_samples,
        evidences_dict,
        valid,
    ):
        self.layers_samples = {}
        self.layers_virtual_samples = {}
        self.dpp_layers = dpp_layers
        self.end_time_tuple = end_time_tuple
        self.num_of_mcmc_samples = num_of_mcmc_samples
        self.evidences_dict = evidences_dict
        self.valid = valid

    def assign_sample(self, layers_samples, layers_virtual_samples, example_ids):
        self.layers_samples = layers_samples
        self.layers_virtual_samples = layers_virtual_samples
        self.example_ids = example_ids

    def calc_stats_from_samples(
        self, jac, virtual_jac, virtual_wrt_real_jac, maximize, parallel=True,
    ):
        initialize_global_vars(
            self,
            jac_init=jac,
            virtual_jac_init=virtual_jac,
            virtual_wrt_real_jac_init=virtual_wrt_real_jac,
            maximize_init=maximize,
            evidences_dict_init=self.evidences_dict,
        )
        l_np_mean = np_mean
        if jac:
            (
                parallel_stats_lst, _, _,
            ) = parallel_ll_grad(self.example_ids, parallel)
        else:
            parallel_stats_lst = parallel_ll_grad(self.example_ids, parallel)

        top_ll = None
        top_mu_grad = None
        top_base_rate = None
        ll = None
        grad = None
        virtual_top_ll = None
        virtual_top_ll_wrt_real = None
        virtual_top_grad_mu = None
        virtual_top_grad_mu_wrt_real = None
        virtual_ll = None
        virtual_ll_wrt_real = None
        virtual_grad = None
        virtual_grad_wrt_real = None
        reduced_var_top_mu_grad = None
        reduced_var_top_virtual_mu_grad = None
        reduced_var_grad = None
        reduced_var_virtual_grad = None
        if maximize:
            # top_ll_lst:  ex_id * sample_id * layer_id
            # top_mu_grad_lst ex_id * sample_id * layer_id
            # top_base_rate_lst  ex_id * sample_id * layer_id
            # ll_dq ex_id * sample_id * layer_id
            # grad_dq ex_id * sample_id * param_id
            # virtual_top_ll_dq  ex_id * sample_id * param_id
            # virtual_top_ll_wrt_real_dq  ex_id * sample_id * param_id
            # virtual_top_grad_mu_dq  ex_id * sample_id * param_id
            # virtual_top_grad_mu_wrt_real_dq ex_id * sample_id * param_id
            # virtual_ll_dq ex_id * sample_id * param_id
            # virtual_ll_wrt_real_dq ex_id * sample_id * param_id
            # virtual_grad_arr  ex_id * sample_id * param_id
            # virtual_grad_wrt_real_arr ex_id * sample_id * param_id
            # reduced_var_top_mu_grad ex_id * sample_id * param_id
            # reduced_var_top_virtual_mu_grad ex_id * sample_id * param_id

            top_base_rate = l_np_mean(
                [s.top_base_rate_lst for s in parallel_stats_lst], axis=1
            )
            top_base_rate = top_base_rate.flatten(order="F")

        top_ll = l_np_mean(
            [s.top_ll_lst for s in parallel_stats_lst], axis=1
        )  # ex_id * layer_id
        # top_ll.flatten(order="F")
        ll = l_np_mean([s.ll_dq for s in parallel_stats_lst], axis=(0, 1))  # layer_id
        virtual_top_ll = l_np_mean(
            [s.virtual_top_ll_dq for s in parallel_stats_lst], axis=1
        )  # ex_id * layer_id
        # virtual_top_ll.flatten(order="F")
        virtual_top_ll_wrt_real = l_np_mean(
            [s.virtual_top_ll_wrt_real_dq for s in parallel_stats_lst], axis=1
        )  #  ex_id * layer_id
        # virtual_top_ll_wrt_real.flatten(order="F")
        virtual_ll = l_np_mean(
            [s.virtual_ll_dq for s in parallel_stats_lst], axis=(0, 1)
        )  # param_id
        virtual_ll_wrt_real = l_np_mean(
            [s.virtual_ll_wrt_real_dq for s in parallel_stats_lst], axis=(0, 1)
        )  # param_id

        if jac:
            top_mu_grad = l_np_mean(
                [s.top_mu_grad_lst for s in parallel_stats_lst], axis=1
            )  # ex_id * layer_id
            top_mu_grad = top_mu_grad.flatten(order="F")
            grad = l_np_mean(
                [s.grad_dq for s in parallel_stats_lst], axis=(0, 1)
            )  # layer_id

        if virtual_jac:
            virtual_top_grad_mu = l_np_mean(
                [s.virtual_top_grad_mu_dq for s in parallel_stats_lst], axis=1
            )  # ex_id * layer_id
            virtual_top_grad_mu = virtual_top_grad_mu.flatten(order="F")
            virtual_grad = l_np_mean(
                [s.virtual_grad_arr for s in parallel_stats_lst], axis=(0, 1)
            )  # param_id
            reduced_var_top_virtual_mu_grad = [
                s.reduced_var_top_virtual_mu_grad for s in parallel_stats_lst
            ]

        if virtual_wrt_real_jac:
            virtual_top_grad_mu_wrt_real = l_np_mean(
                [s.virtual_top_grad_mu_wrt_real_dq for s in parallel_stats_lst], axis=1
            )  # ex_id * layer_id
            virtual_top_grad_mu_wrt_real = virtual_top_grad_mu_wrt_real.flatten(
                order="F"
            )
            virtual_grad_wrt_real = l_np_mean(
                [s.virtual_grad_wrt_real_arr for s in parallel_stats_lst], axis=(0, 1)
            )  # param_id

        stats = Stats_collections(
            top_ll=top_ll,
            top_mu_grad=top_mu_grad,
            top_base_rate=top_base_rate,
            ll=ll,
            grad=grad,
            virtual_top_ll=virtual_top_ll,
            virtual_top_ll_wrt_real=virtual_top_ll_wrt_real,
            virtual_top_grad_mu=virtual_top_grad_mu,
            virtual_top_grad_mu_wrt_real=virtual_top_grad_mu_wrt_real,
            virtual_ll=virtual_ll,
            virtual_ll_wrt_real=virtual_ll_wrt_real,
            virtual_grad=virtual_grad,
            virtual_grad_wrt_real=virtual_grad_wrt_real,
            reduced_var_top_mu_grad=reduced_var_top_mu_grad,
            reduced_var_top_virtual_mu_grad=reduced_var_top_virtual_mu_grad,
            reduced_var_grad=reduced_var_grad,
            reduced_var_virtual_grad=reduced_var_virtual_grad,
        )

        return stats

    def check_ll_grad(
        self, example_ids, jac, virtual_jac, virtual_wrt_real_jac, maximize
    ):
        x0_top_mu = self.dpp_layers.get_top_mu(example_ids, valid=self.valid)
        x0_top_virtual_mu = self.dpp_layers.get_top_virtual_mu(
            example_ids, valid=self.valid
        )
        x0_real_kernel_params = self.dpp_layers.get_real_kernel_params()
        x0_virtual_kernel_params = self.dpp_layers.get_virtual_params()
        x0_stats = self.calc_stats_from_samples(
            jac, virtual_jac, virtual_wrt_real_jac, maximize,
        )

        # check maximized top base rate; the grad should be 0
        print("check top base rate")
        self.dpp_layers.set_top_mu(
            x0_stats.top_base_rate, example_ids, valid=self.valid, transform=False
        )
        stats = self.calc_stats_from_samples(
            jac, virtual_jac, virtual_wrt_real_jac, maximize=False
        )
        # print("ana grad for maximized top base rate = ", stats.top_mu_grad)
        self.dpp_layers.set_top_mu(
            x0_top_mu, example_ids, valid=self.valid, transform=True
        )
        if np_sum(np_abs(stats.top_mu_grad)) > 1e-10:
            print("ana grad for maximized top base rate = ", stats.top_mu_grad)

        # check top base rate grad
        print("check top mu grad")
        fin_diff = np.zeros_like(x0_top_mu)
        x0_stats_top_ll_flatten = x0_stats.top_ll.flatten(order="F")
        for f in range(len(fin_diff)):
            x_new = np.array(x0_top_mu)
            x_new[f] += 1e-9
            self.dpp_layers.set_top_mu(x_new, example_ids, valid=self.valid)
            stats = self.calc_stats_from_samples(
                jac, virtual_jac, virtual_wrt_real_jac, maximize=False,
            )
            stats_top_ll = stats.top_ll.flatten(order="F")
            fin_diff[f] = (stats_top_ll[f] - x0_stats_top_ll_flatten[f]) / 1e-9
        # print(f"finite diff = {fin_diff}, analytical diff = {x0_stats.top_mu_grad}")
        if np_sum(np_abs(fin_diff - x0_stats.top_mu_grad)) > 1e-6:
            print(f"finite diff = {fin_diff}, analytical diff = {x0_stats.top_mu_grad}")
        self.dpp_layers.set_top_mu(
            x0_top_mu, example_ids, valid=self.valid, transform=True
        )

        # check top virtual base rate grad
        print("check top virtual base rate grad")
        print("virtual grad mu", np_mean(x0_stats.virtual_top_grad_mu))
        fin_diff = np.zeros_like(x0_top_virtual_mu)
        x0_stats_virtual_top_ll_flatten = x0_stats.virtual_top_ll.flatten(order="F")
        for f in range(len(fin_diff)):
            x_new = np.array(x0_top_virtual_mu)
            x_new[f] += 1e-9
            self.dpp_layers.set_top_virtual_mu(x_new, example_ids, valid=self.valid)
            stats = self.calc_stats_from_samples(
                jac, virtual_jac, virtual_wrt_real_jac, maximize=False,
            )
            stats_virtual_top_ll = stats.virtual_top_ll.flatten(order="F")
            fin_diff[f] = (
                stats_virtual_top_ll[f] - x0_stats_virtual_top_ll_flatten[f]
            ) / 1e-9
        # print(
        #     f"finite diff = {fin_diff}, analytical diff = {x0_stats.virtual_top_grad_mu}"
        # )
        if np_sum(np_abs(fin_diff - x0_stats.virtual_top_grad_mu)) > 1e-6:
            print(
                f"finite diff = {fin_diff}, analytical diff = {x0_stats.virtual_top_grad_mu}"
            )
        self.dpp_layers.set_top_virtual_mu(
            x0_top_virtual_mu, example_ids, valid=self.valid
        )

        # check top virtual base rate wrt real grad
        print("check top virtual base rate wrt real grad")
        fin_diff = np.zeros_like(x0_top_virtual_mu)
        x0_stats_virtual_top_ll_wrt_real_flatten = x0_stats.virtual_top_ll_wrt_real.flatten(
            order="F"
        )
        for f in range(len(fin_diff)):
            x_new = np.array(x0_top_virtual_mu)
            x_new[f] += 1e-9
            self.dpp_layers.set_top_virtual_mu(x_new, example_ids, valid=self.valid)
            stats = self.calc_stats_from_samples(
                jac, virtual_jac, virtual_wrt_real_jac, maximize=False
            )
            stats_virtual_top_ll_wrt_real = stats.virtual_top_ll_wrt_real.flatten(
                order="F"
            )
            fin_diff[f] = (
                stats_virtual_top_ll_wrt_real[f]
                - x0_stats_virtual_top_ll_wrt_real_flatten[f]
            ) / 1e-9
        print(
            f"finite diff = {fin_diff}, analytical diff = {x0_stats.virtual_top_grad_mu_wrt_real}"
        )
        self.dpp_layers.set_top_virtual_mu(
            x0_top_virtual_mu, example_ids, valid=self.valid
        )

        # check real kernel params grad
        print("check real kernel params grad")
        ll_tot = np_sum(np_mean(x0_stats.top_ll, axis=0)) + np_sum(x0_stats.ll)
        print("ll tot = ", ll_tot)
        fin_diff = np.zeros_like(x0_real_kernel_params)
        for f in range(len(fin_diff)):
            x_new = np.array(x0_real_kernel_params)
            x_new[f] += 1e-9
            self.dpp_layers.set_real_kernel_params(x_new)
            stats = self.calc_stats_from_samples(
                jac, virtual_jac, virtual_wrt_real_jac, maximize=False,
            )
            fin_diff[f] = (np_sum(stats.ll) - np_sum(x0_stats.ll)) / 1e-9
        print(f"finite diff = {fin_diff}, analytical diff = {x0_stats.grad}")
        self.dpp_layers.set_real_kernel_params(x0_real_kernel_params,)

        # check virtual kernel params grad
        print("check virtual kernel params grad")
        virtual_ll_tot = np_sum(np_mean(x0_stats.virtual_top_ll, axis=0)) + np_sum(
            x0_stats.virtual_ll
        )
        print("virtual ll tot = ", virtual_ll_tot)
        fin_diff = np.zeros_like(x0_virtual_kernel_params)
        for f in range(len(fin_diff)):
            x_new = np.array(x0_virtual_kernel_params)
            x_new[f] += 1e-9
            self.dpp_layers.set_virtual_params(x_new,)
            stats = self.calc_stats_from_samples(
                jac, virtual_jac, virtual_wrt_real_jac, maximize=False,
            )
            # if f < len(virtual_var_ids) * len(self.dpp_layers.top_ids_lst) * len(
            #     self.dpp_layers.children_ids_lst[0]
            # ) and f % len(virtual_var_ids) != 0 :
            #     fin_diff[f] = (
            #         np_sum(np_mean(stats.virtual_top_ll, axis=0))
            #         - np_sum(np_mean(x0_stats.virtual_top_ll, axis=0))
            #     ) / 1e-8
            # else:
            #     fin_diff[f] = (
            #         np_sum(stats.virtual_ll) - np_sum(x0_stats.virtual_ll)
            #     ) / 1e-8
            fin_diff[f] = (
                np_sum(np_mean(stats.virtual_top_ll, axis=0))
                + np_sum(stats.virtual_ll)
                - np_sum(np_mean(x0_stats.virtual_top_ll, axis=0))
                - np_sum(x0_stats.virtual_ll)
            ) / 1e-9
        print(f"finite diff = {fin_diff}, analytical diff = {x0_stats.virtual_grad}")
        self.dpp_layers.set_virtual_params(x0_virtual_kernel_params)

        # check virtual kernel params grad wrt real
        print("check virtual kernel params grad wrt real")
        fin_diff = np.zeros_like(x0_virtual_kernel_params)
        for f in range(len(fin_diff)):
            x_new = np.array(x0_virtual_kernel_params)
            x_new[f] += 1e-9
            self.dpp_layers.set_virtual_params(x_new,)
            stats = self.calc_stats_from_samples(
                jac, virtual_jac, virtual_wrt_real_jac, maximize=False,
            )
            # if (
            #     f
            #     < len(virtual_var_ids)
            #     * len(self.dpp_layers.top_ids_lst)
            #     * len(self.dpp_layers.children_ids_lst[0])
            #     and f % len(virtual_var_ids) != 0
            # ):
            #     fin_diff[f] = (
            #         np_sum(np_mean(stats.virtual_top_ll_wrt_real, axis=0))
            #         - np_sum(np_mean(x0_stats.virtual_top_ll_wrt_real, axis=0))
            #     ) / 1e-9
            # else:
            #     fin_diff[f] = (
            #         np_sum(stats.virtual_ll_wrt_real)
            #         - np_sum(x0_stats.virtual_ll_wrt_real)
            #     ) / 1e-9
            fin_diff[f] = (
                np_sum(np_mean(stats.virtual_top_ll_wrt_real, axis=0))
                + np_sum(stats.virtual_ll_wrt_real)
                - np_sum(np_mean(x0_stats.virtual_top_ll_wrt_real, axis=0))
                - np_sum(x0_stats.virtual_ll_wrt_real)
            ) / 1e-9
        print(
            f"finite diff = {fin_diff}, analytical diff = {x0_stats.virtual_grad_wrt_real}"
        )
        self.dpp_layers.set_virtual_params(x0_virtual_kernel_params)
