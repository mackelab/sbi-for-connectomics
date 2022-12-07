import time
from typing import Callable, Tuple, Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from .utils import *


class DistanceRuleSimulator:
    """Simulate Peters' rule with the dense structural model.

    We define new features, e.g., number of common cubes for each neuron pairs, to
    mimic Peters' rule and perform inference on it.

    The new features are loaded from files which filenames are hardcoded below.

    We use the same literature in vivo measurements as xo as before, their IDs
    are also hardcoded below.
    """

    def __init__(
        self,
        path_to_model,
        num_subsampling_pairs: int = -1,
        cube_size: int = 25,
        feature_set_name: str = "set-7",
    ) -> None:

        # Hard coded magic strings.
        print(f"using feature {feature_set_name}")
        # feature_set_name = (
        #     # "set-7"  # holds subcellular level features: 1 if two neurons meet in a voxel, 0 else.
        #     "set-6"  # Set 6 holds the common cubes for different cube sizes.
        #     # "set-5"  # Set 5 holds common cubes, min distance and axon-dendrite product for cube size 50mu-m
        # )
        literature_population_ids = [
            "897f7fba-714c-4e2e-9f9c-552e3815fb34",
            "8bbcab58-cd1e-48ac-b7ef-583e8a077a9f",
            "7d6ff52f-0810-4ff8-8bad-e41173c5d2ff",
            "1c32cfac-828a-46ce-acaa-8de786e6c22f",
            "1d0687e0-bb9e-4cb1-9ed5-fcc00cd29c4d",
            "fe21fc12-9bda-45c9-857a-385b418a1116",
            "2eab5e02-e8da-485e-8989-41554deae0c1",
        ]
        cube_sizes = [50, 25, 10, 5, 1]
        self.num_subsampling_pairs = num_subsampling_pairs

        # Load features
        path_to_features = os.path.join(path_to_model, "features")
        path_to_mask = os.path.join(path_to_model, "masks")
        features = np.loadtxt(
            path_to_features + f"/features_{feature_set_name}_float.tsv",
            dtype=np.float32,
        )
        if feature_set_name == "set-6":
            assert cube_size in cube_sizes
            feature_mask = np.array(cube_sizes) == cube_size
            self.common_cubes = features[:, feature_mask]
        elif feature_set_name == "set-7":
            self.meet_in_cube = features

        else:
            self.common_cubes = features[:, 0]
            self.min_distance = features[:, 1]
            # Scale from mu-m to mm
            self.axon_dendrite_length_mm = features[:, 2] / 1000

        self.all_ids = np.loadtxt(
            path_to_features + f"/features_{feature_set_name}_int.tsv", dtype=np.int32
        )
        self.neuron_pair_ids = np.zeros(self.all_ids.shape[0], dtype=np.int64)
        for idx, id in enumerate(self.all_ids):
            self.neuron_pair_ids[idx] = int(str(id[1]) + str(id[2]))
        _, unique_idx = np.unique(self.neuron_pair_ids, return_index=True)
        self.unique_pair_ids = self.neuron_pair_ids[unique_idx]

        # Load literature info for all literature data
        publications = loadPublications(
            os.path.join(path_to_model, "empirical_data_2020-07-13.json"),
            extend_sort=True,
        )
        # Keep only the seven in vivo publications
        self.publications = [
            p for p in publications if p["ID"] in literature_population_ids
        ]
        self.paper_titles = [
            p for p in publications if p["title"] in literature_population_ids
        ]

        # Load masks for populations used in those publications.
        self.population_masks = []
        self.unique_pairs_per_pop = []
        for publication in self.publications:
            maskFile = os.path.join(
                path_to_mask,
                "mask_{}_{}.txt".format(feature_set_name, publication["ID"]),
            )
            mask = np.loadtxt(maskFile, dtype="uint32")
            self.population_masks.append(mask)
            _, unique_idx = np.unique(self.neuron_pair_ids[mask], return_index=True)
            ids = self.neuron_pair_ids[mask][unique_idx]
            self.unique_pairs_per_pop.append(ids)

        # Norm cube counts for each population for proximity cutoff.
        if not feature_set_name == "set-7":
            self.common_cubes_normed_per_pop = self.norm_cube_counts_per_pop()
            self.common_cubes_normed_global = (
                self.common_cubes / self.common_cubes.max()
            )

    def norm_cube_counts_per_pop(self):
        """Return common cube counts max-normed within each population."""

        normed_cube_counts = np.zeros_like(self.common_cubes)
        for pop_mask in self.population_masks:
            normed_cube_counts[pop_mask] = (
                self.common_cubes[pop_mask] / self.common_cubes[pop_mask].max()
            )

        return normed_cube_counts

    def rule(self, theta: Tensor, feature: ndarray, connection_fun: callable):
        """Return connection probabilities according the Peters' rule based on the
        number of common cubes.

        Details:
            - subsample before sampling,
            - apply rule,
            - calculate connection probabilities directly in this function
        """

        batch_size, D = theta.shape
        assert D < 3
        theta = theta.numpy()

        probs = np.zeros((batch_size, len(self.population_masks)))

        # Iterate over batch.
        for batch_idx in range(batch_size):

            # For each parameter, iterate over population pairs.
            for pop_idx, pop_mask in enumerate(self.population_masks):
                # We need to resample for every prob and every parameter.
                if not self.num_subsampling_pairs == -1:
                    subsampling_idxs = np.random.randint(
                        low=0, high=len(pop_mask), size=self.num_subsampling_pairs
                    )
                else:
                    # Select all pairs in mask.
                    subsampling_idxs = np.arange(len(pop_mask))

                # Use predefined function to calculate connections.
                cij = connection_fun(
                    theta[
                        batch_idx,
                    ],
                    feature[pop_mask][subsampling_idxs],
                )

                # Take mean over selected pairs.
                probs[batch_idx, pop_idx] = cij.mean()

        return torch.tensor(probs, dtype=torch.float32)

    def rule_subcellular(self, theta: Tensor) -> Tensor:

        batch_size, D = theta.shape
        assert D < 3
        theta = theta.numpy()

        probs = np.zeros((batch_size, len(self.population_masks)))
        # Sample with probability theta for each ijk where ij meet (feature == 1)
        # Feature is 0 wherever they do not meet.
        synapses = np.random.binomial(n=1, p=theta * self.meet_in_cube)

        # Iterate over batch.
        for batch_idx in range(batch_size):

            # Calculate connection probabilities per population.
            for pop_idx, pop_mask in enumerate(self.population_masks):

                pop_synapses = synapses[batch_idx, pop_mask]
                pop_unique_ids = self.unique_pairs_per_pop[pop_idx]

                # Calculate connections per neuron pair
                cij = np.zeros(pop_unique_ids.size)
                for idx, pair_id in enumerate(pop_unique_ids):
                    cij[idx] = (
                        1
                        if pop_synapses[self.neuron_pair_ids[pop_mask] == pair_id].sum()
                        > 0
                        else 0
                    )

                probs[batch_idx, pop_idx] = cij.mean()

        return torch.tensor(probs, dtype=torch.float32)

    @staticmethod
    def bernoulli_glm(
        theta: ndarray,
        feature: ndarray,
    ) -> ndarray:
        """Return connections sampled through Bernoulli GLM."""

        assert theta.shape == (2,)
        pij = 1 / (1 + np.exp(-(theta[1] * feature + theta[0])))

        # Sample connections for all selected pairs
        return np.random.binomial(n=1, p=pij)

    @staticmethod
    def cutoff_rule(
        theta: float,
        feature: ndarray,
    ) -> ndarray:
        cij = np.zeros_like(feature)
        cij[feature >= theta] = 1
        return cij


class RuleSimulator:
    """Applies a rule to a dense structural model to simulate synapse formation.

    The rule is defined as callable with the right number of of parameters as
    needed for the structural model, e.g., three for the default rule.

    The structural model is passed via a file path to the model data from ZIB.
    """

    def __init__(
        self,
        path_to_model: str,
        rule: Callable,
        return_synapse_counts: bool = False,
        experiment_name: str = "default_rule_invivo",
        max_rate: int = 10,
        verbose: bool = False,
        num_subsampling_pairs: int = -1,
        prelocate_postall_offset: bool = False,
    ) -> None:
        """Create rule simulator by passing the path to the model data and the rule.

        Args:
            path_to_model: absolute path to the model files.
            rule: python function defining the mapping from parameters to
                synapse counts.
            return_synapse_counts: Whether to return raw synapse counts or
                connection probs (False).
            experiment_name: name of the experiment.
            max_rate: upper bound for poisson rate.
            verbose: Boolean for extended print statements (False).
            num_subsampling_pairs: Number of pairs to subsample when calculating
                population connection probability across pairs. Default = -1, no
                subsampling.
        """

        literature_population_ids = [
            "897f7fba-714c-4e2e-9f9c-552e3815fb34",
            "8bbcab58-cd1e-48ac-b7ef-583e8a077a9f",
            "7d6ff52f-0810-4ff8-8bad-e41173c5d2ff",
            "1c32cfac-828a-46ce-acaa-8de786e6c22f",
            "1d0687e0-bb9e-4cb1-9ed5-fcc00cd29c4d",
            "fe21fc12-9bda-45c9-857a-385b418a1116",
            "2eab5e02-e8da-485e-8989-41554deae0c1",
        ]

        self.path_to_model = path_to_model
        self.verbose = verbose
        self.rule = rule
        # Upper bound for poisson rate.
        self.max_rate = max_rate
        # Sample of postsynaptic neurons (to emulate sample size of experiments)
        self.num_subsampling_pairs = num_subsampling_pairs

        # Load experiment meta data from file.
        experiment_spec = getInferenceExperimentSpec(
            experiments=loadJsonData(
                os.path.join(path_to_model, "inference_experiments_local.json")
            ),
            identifier=experiment_name,
        )
        self.feature_set = experiment_spec["feature-set"]

        # load empirical data from literatur to build summary statistics masks.
        publications = loadPublications(
            os.path.join(path_to_model, "empirical_data_2020-07-13.json"),
            extend_sort=True,
        )
        self.publications = [
            p for p in publications if p["ID"] in literature_population_ids
        ]
        self.population_masks = []
        for publication in self.publications:
            maskFile = os.path.join(
                path_to_model,
                "masks",
                "mask_{}_{}.txt".format(self.feature_set, publication["ID"]),
            )
            mask = np.loadtxt(maskFile, dtype="uint32")
            self.population_masks.append(mask)

        constraints = getConstraints(
            self.publications, self.feature_set, os.path.join(path_to_model, "masks")
        )

        # Load required features from model data files.
        features_int, features_float = self.load_features()

        self.features_int = features_int
        self.features_float = features_float
        self.features_float_log = self.get_log_features(features_float)

        self.constraints = constraints
        self.n_constraints = len(constraints["masks"])
        # Pre locate features and ids per constraint.
        (
            self.features_int_per_constraint,
            self.pre_ids_per_constraint,
            self.post_ids_per_constraint,
            self.pre_matrix_idx_per_constraint,
            self.post_matrix_idx_per_constraint,
        ) = self.prelocate_per_constraint()

        # Prelocate indices for calculating postall from post.
        self.prelocate_post_sum_idxs()

        if prelocate_postall_offset:
            self.global_norm = self.features_float[:, 2].mean()
            self.log_global_norm = np.log(self.global_norm)
            self.postall_offset = self.get_postall_offset()

            self.prelocate_property_masks()
            self.assert_property_masks_for_rules()
        self.return_synapse_counts = return_synapse_counts

    def get_postall_offset(self) -> ndarray:

        postall_hat = self.sum_post_across_voxels(self.features_float[:, 1])

        # The difference gives the offset.
        return self.features_float[:, 2] - postall_hat

    def prelocate_post_sum_idxs(self) -> None:
        """Set indices for summing over post features to get postall."""

        postids = self.features_int[:, 2]
        voxelids = self.features_int[:, 3]

        # Find relevant voxelids by taking only unique postid-voxelid pairs
        _, idx1, inv1 = np.unique(
            np.hstack((postids.reshape(-1, 1), voxelids.reshape(-1, 1))),
            axis=0,
            return_index=True,
            return_inverse=True,
        )
        relevant_ids = voxelids[idx1]

        # Find unique voxel ids among the relevant ones
        uvids, inv2 = np.unique(relevant_ids, return_inverse=True)

        # For each unique voxel id collect indices where to find those ids
        vidxs = []
        for vid in uvids:
            vidxs.append(np.where(relevant_ids == vid)[0])

        # the indices to get the unique representation of post.
        # the summing indices
        # and the indices for the inverse unique operations
        self.unique_post_postall_idxs = idx1
        self.post_sum_idxs = vidxs
        self.iunique_to_post_postall = inv2
        self.iunique_to_all = inv1
        self.empty_postall_idxs = np.where(voxelids == -1)[0]

    def sum_post_across_voxels(self, post: ndarray) -> ndarray:
        """Given a post feature vector, return the postall feature vector.

        Postall is obtained from post by summing only the relevant entries,
        e.g., leaving out duplicates due to multiple post neurons per voxel
        and due to multiple pre neurons per post neuron.

        The relevant indices are prelocated as masks in self.
        """
        relevant_post = post[self.unique_post_postall_idxs]

        sums = []
        for idx in self.post_sum_idxs:
            sums.append(relevant_post[idx].sum())
        sums = np.array(sums)

        # Return one sum for each voxelid
        # Apply inverse unique operations twice,
        # first for going from unique voxels to unique postid-voxel
        # second for going from there to all rows, including duplicate due to multiple pre neurons per post neuron.
        return sums[self.iunique_to_post_postall][self.iunique_to_all]

    def postall_from_post(self, post: ndarray) -> ndarray:
        """Return postall calculated from post and offset."""
        post_sum = self.sum_post_across_voxels(post)
        postall = np.round(post_sum + self.postall_offset, decimals=1)
        # Set -1 indices to value as in actual postall
        postall[self.empty_postall_idxs] = 1e-10

        return postall

    def prelocate_property_masks(self):
        """Return masks for specific properties (e.g., cell type) as needed by some rules."""

        # Single property predicates:
        # Layer4 (granular)
        is_l4 = self.features_int[:, 4] == 2
        not_l4 = ~is_l4
        # infragranular
        is_infra = self.features_int[:, 4] == 1
        not_infra = ~is_infra
        # L4ss, potentially with structures outside Layer4.
        is_l4ss = self.features_int[:, 6] == 4
        not_l4ss = ~is_l4ss

        is_l4sp = self.features_int[:, 6] == 3
        not_l4sp = ~is_l4sp

        is_l5it = self.features_int[:, 6] == 5
        not_l5it = ~is_l5it

        is_l5pt = self.features_int[:, 6] == 6
        not_l5pt = ~is_l5pt

        # Combined property predicates:
        is_l4l4ss = is_l4 & is_l4ss
        is_l4rest = is_l4 & ~is_l4ss
        not_4lss_l4sp_l5it_l5pt = not_l4ss & not_l4sp & not_l5it & not_l5pt
        not_l4ss_l4sp_l5it_l5pt_infra = not_4lss_l4sp_l5it_l5pt & is_infra
        not_l4ss_l4sp_l5it_l5pt_not_infra = not_4lss_l4sp_l5it_l5pt & not_infra

        self.is_l4 = is_l4
        self.not_l4 = not_l4
        self.is_l4ss = is_l4ss
        self.not_l4ss = not_l4ss
        self.is_l4l4ss = is_l4l4ss
        self.is_l4rest = is_l4rest
        self.is_l4sp = is_l4sp
        self.is_l5it = is_l5it
        self.is_l5pt = is_l5pt
        self.not_4lss_l4sp_l5it_l5pt = not_4lss_l4sp_l5it_l5pt
        self.not_l4ss_l4sp_l5it_l5pt_infra = not_l4ss_l4sp_l5it_l5pt_infra
        self.not_l4ss_l4sp_l5it_l5pt_not_infra = not_l4ss_l4sp_l5it_l5pt_not_infra

    def assert_property_masks_for_rules(self):
        """Assert that property masks are consistent for all rules."""

        self.assert_property_masks([self.is_l4, self.not_l4], "rule_Layer4")
        self.assert_property_masks([self.is_l4ss, self.not_l4ss], "rule_L4ss")
        self.assert_property_masks(
            [self.is_l4l4ss, self.is_l4rest, self.not_l4], "rule_l4_l4ss_rest"
        )
        self.assert_property_masks(
            [
                self.is_l4ss,
                self.is_l4sp,
                self.is_l5it,
                self.is_l5pt,
                self.not_4lss_l4sp_l5it_l5pt,
            ],
            "rule_L4ss_L4sp_L5it_l5pt_linear",
        )
        self.assert_property_masks(
            [
                self.is_l4ss,
                self.is_l4sp,
                self.is_l5it,
                self.is_l5pt,
                self.not_l4ss_l4sp_l5it_l5pt_not_infra,
                self.not_l4ss_l4sp_l5it_l5pt_infra,
            ],
            "rule_L4ss_L4sp_L5it_l5pt_infra_linear",
        )

        # print("property mask: is_l4ss", np.count_nonzero(self.is_l4ss))
        # print("property mask: is_l4sp", np.count_nonzero(self.is_l4sp))
        # print("property mask: is_l5it", np.count_nonzero(self.is_l5it))
        # print("property mask: is_l5pt", np.count_nonzero(self.is_l5pt))

    def assert_property_masks(self, masks, ruleName):
        """Assert that property masks are disjoint and cover all rows of feature vector."""

        nRows = self.features_float.shape[0]
        nRowsMasks = 0
        unionedMask = np.zeros(nRows, dtype=bool)
        for mask in masks:
            unionedMask |= mask
            nRowsMasks += np.count_nonzero(mask)

        assert nRows == np.count_nonzero(
            unionedMask
        ), "Property masks do not cover all rows ({}).".format(ruleName)
        assert nRowsMasks == nRows, "Property masks are not disjoint ({}).".format(
            ruleName
        )

    def prelocate_per_constraint(self):

        features_int_per_constraint = []
        pre_ids_per_constraint = []
        post_ids_per_constraint = []
        pre_matrix_idx_per_constraint = []
        post_matrix_idx_per_constraint = []

        for i in range(0, self.n_constraints):
            features_int = self.features_int[self.constraints["masks"][i], :]
            features_int_per_constraint.append(features_int)
            pre_ids = np.unique(features_int[:, 1])
            post_ids = np.unique(features_int[:, 2])
            pre_matrix = self.prelocate_matrix_idx(features_int[:, 1], pre_ids)
            pre_matrix_idx_per_constraint.append(pre_matrix)
            post_matrix = self.prelocate_matrix_idx(features_int[:, 2], post_ids)
            post_matrix_idx_per_constraint.append(post_matrix)
            pre_ids_per_constraint.append(pre_ids)
            post_ids_per_constraint.append(post_ids)

            noOverlap = np.count_nonzero(features_int[:, 3] == -1)
            if self.verbose:
                print(
                    "constraint {}, num rows (with overlap) {}".format(
                        i, features_int.shape[0] - noOverlap
                    )
                )

        return (
            features_int_per_constraint,
            pre_ids_per_constraint,
            post_ids_per_constraint,
            pre_matrix_idx_per_constraint,
            post_matrix_idx_per_constraint,
        )

    def prelocate_matrix_idx(self, ids_feature_vector, ids_unique):
        """Precompute the index mapping: featureVector-row --> connectivity matrix (row/col: pre/post)"""

        matrix_idx = np.zeros(ids_feature_vector.shape[0], dtype=int)
        for i in range(0, ids_feature_vector.shape[0]):
            matrix_idx[i] = np.where(ids_unique == ids_feature_vector[i])[0]
        return matrix_idx

    def __call__(self, theta: Union[ndarray, Tensor]) -> Tensor:
        """Apply rule parametrized by theta and return summary stats (or counts)."""

        theta_numpy = self.process_theta(theta)

        # startSynapses = time.time()
        synapses = self.rule(model=self, theta=theta_numpy)
        # print(f"Compute synapses: {time.time() - startSynapses}")

        # Maybe compute summary statistics.
        if self.return_synapse_counts:
            return torch.as_tensor(synapses, dtype=torch.float32)
        else:
            # startStats = time.time()
            summary_matrix = np.zeros(len(self.constraints["masks"]))
            for c_idx in range(0, self.n_constraints):
                features_int_i = self.features_int_per_constraint[c_idx]
                # post_ids_i = self.post_ids_per_constraint[c_idx],
                pre_matrix_idx_i = self.pre_matrix_idx_per_constraint[c_idx]
                post_matrix_idx_i = self.post_matrix_idx_per_constraint[c_idx]
                synapses_i = synapses[self.constraints["masks"][c_idx]]

                # Get indices of non-zero counts
                non_zero_idx = np.where(synapses_i > 0)[0]

                # If there are no counts at all, all probs are zero.
                if len(non_zero_idx) == 0:
                    continue

                if features_int_i[0, 0] == 0:
                    # invivo connection probability
                    if self.num_subsampling_pairs == -1:
                        summary_matrix[c_idx] = self.calculate_connection_probability(
                            pre_ids=self.pre_ids_per_constraint[c_idx],
                            post_ids=self.post_ids_per_constraint[c_idx],
                            pre_matrix_idx=pre_matrix_idx_i[non_zero_idx],
                            post_matrix_idx=post_matrix_idx_i[non_zero_idx],
                        )
                    else:
                        summary_matrix[
                            c_idx
                        ] = self.calculate_connection_probability_subsampled(
                            pre_ids=self.pre_ids_per_constraint[c_idx],
                            post_ids=self.post_ids_per_constraint[c_idx],
                            pre_matrix_idx=pre_matrix_idx_i[non_zero_idx],
                            post_matrix_idx=post_matrix_idx_i[non_zero_idx],
                        )
                else:
                    # invitro connection probability
                    if self.num_subsampling_pairs == -1:
                        summary_matrix[
                            c_idx
                        ] = self.calculate_connection_probability_slice(
                            features_int_i, non_zero_idx
                        )
                    else:
                        raise NotImplementedError
            # print(f"Compute stats: {time.time() - startStats}")
            return torch.as_tensor(summary_matrix, dtype=torch.float32).unsqueeze(0)

    def process_theta(self, theta: Union[ndarray, Tensor]) -> ndarray:

        if theta.ndim > 1:
            assert (
                theta.shape[0] == 1
            ), "rule simulator simulates single parameters only."

        if isinstance(theta, Tensor):
            theta_numpy = theta.numpy()
        else:
            theta_numpy = theta

        return theta_numpy.squeeze()

    def load_features(self) -> Tuple[ndarray, ndarray]:
        """Return features loaded from file for given model and constraints."""

        start = time.time()
        filenameInt = os.path.join(
            self.path_to_model,
            "features",
            "features_{}_int.tsv".format(self.feature_set),
        )
        filenameFloat = os.path.join(
            self.path_to_model,
            "features",
            "features_{}_float.tsv".format(self.feature_set),
        )

        if self.verbose:
            print("Loading features, this may take a while...")
        features_int = np.loadtxt(filenameInt, dtype=np.int32)
        features_float = np.loadtxt(filenameFloat, dtype=np.float32)

        if self.verbose:
            print(f"Time elapsed: {time.time() - start}")

        if features_float.ndim == 1:
            features_float = features_float.reshape(-1, 1)

        return np.atleast_2d(features_int), np.atleast_2d(features_float)

    def get_log_features(self, features):

        # Set zero depth to small value to avoid log-nan.
        features[features == 0] = 1e-10

        # -1 coding for missing cube will become NaN.
        log_features = np.log(features)

        # The actual features must be finite.
        assert np.isfinite(log_features[:, :3]).all(), "log features must be finite."

        return log_features

    def calculate_connection_probability(
        self, pre_ids, post_ids, pre_matrix_idx, post_matrix_idx
    ):
        """Return connection probability as proportion of present connection."""

        # build 0-1 connectivity matrix from calculated connected indices.
        cellularConnectivityMatrix = np.zeros((pre_ids.size, post_ids.size))
        cellularConnectivityMatrix[pre_matrix_idx, post_matrix_idx] = 1

        # Connection probability is given by the mean over all population pairs.
        return cellularConnectivityMatrix.mean()

    def calculate_connection_probability_subsampled(
        self, pre_ids, post_ids, pre_matrix_idx, post_matrix_idx
    ):
        """Return population connection probability determined from subset of neurons.

        The subset is sampled randomly in the pre and post population neurons to
        mimick experimental conditions.
        """
        num_pre = pre_ids.size
        num_post = post_ids.size

        # build 0-1 connectivity matrix from calculated connected indices.
        cellularConnectivityMatrix = np.zeros((num_pre, num_post))
        cellularConnectivityMatrix[pre_matrix_idx, post_matrix_idx] = 1

        # Sample random indices for each dimension.
        pre_ids_sample = np.random.randint(
            0, high=num_pre, size=((self.num_subsampling_pairs,))
        )
        post_ids_sample = np.random.randint(
            0, high=num_post, size=((self.num_subsampling_pairs,))
        )
        # Select the pair entries from the matrix and calculate proportion connected.
        prob = cellularConnectivityMatrix[pre_ids_sample, post_ids_sample].mean()
        return prob

    def calculate_connection_probability_slice(self, features_int, non_zero_idx):

        synaptic_connections = features_int[non_zero_idx]
        cellularConnections = {}
        numRealized = {}
        numPost = {}
        for i in range(0, len(features_int)):
            preId = features_int[i, 1]
            postId = features_int[i, 2]
            cellularConnections[(preId, postId)] = 0
            if preId not in numPost.keys():
                numPost[preId] = set()
            numPost[preId].add(postId)
            numRealized[preId] = 0
        for i in range(0, synaptic_connections.shape[0]):
            preId = synaptic_connections[i, 1]
            postId = synaptic_connections[i, 2]
            cellularConnections[(preId, postId)] = 1
        for preId_postId, realized in cellularConnections.items():
            numRealized[preId_postId[0]] += realized
        fractionRealized = []
        for preId, numRealized in numRealized.items():
            prob = numRealized / len(numPost[preId])
            fractionRealized.append(prob)
        prob = np.mean(fractionRealized)
        return prob

    def get_x_o(self) -> Tensor:
        return torch.as_tensor(self.constraints["observables"], dtype=torch.float32)


def draw_synapses_from_poisson(model: RuleSimulator, rate: ndarray):
    """Return synapse counts from poisson given Poisson rate."""

    synapses = np.random.poisson(lam=rate)
    # Set counts of pairs without overlap (-1) to zero.
    synapses[np.where(model.empty_postall_idxs)] = 0

    return synapses


def global_norm_rule(model: RuleSimulator, theta: ndarray):
    """Rule scaling just pre and post with global instead of voxel specific
    normalization."""

    log_dso = (
        theta * (model.features_float_log[:, 0] + model.features_float_log[:, 1])
        - model.log_global_norm
    )

    # Clip rate from above and below.
    clipped_dso = np.clip(log_dso, -model.max_rate, model.max_rate)
    return draw_synapses_from_poisson(model, rate=np.exp(clipped_dso))


def one_param_rule(model: RuleSimulator, theta: ndarray):
    """Scale the entire DSO."""

    log_dso = theta * (
        model.features_float_log[:, 0]
        + model.features_float_log[:, 1]
        - model.features_float_log[:, 2]
    )
    # Clip rate from above and below.
    clipped_dso = np.clip(log_dso, -model.max_rate, model.max_rate)
    return draw_synapses_from_poisson(model, rate=np.exp(clipped_dso))


def two_param_rule(model: RuleSimulator, theta: ndarray):
    """Scale pre*post and postall seperately."""

    assert theta.shape == (2,), "This is a two-parameter rule."

    log_dso = (
        theta[0] * (model.features_float_log[:, 0] + model.features_float_log[:, 1])
        - theta[1] * model.features_float_log[:, 2]
    )
    # Clip rate from above and below.
    clipped_dso = np.clip(log_dso, -model.max_rate, model.max_rate)
    return draw_synapses_from_poisson(model, rate=np.exp(clipped_dso))


def two_param_rule_dependent(model: RuleSimulator, theta: ndarray, offset: float=3.0):
    """Scale pre*post and postall seperately."""

    assert theta.shape == (2,), "This is a two-parameter rule."

    log_dso = (
        theta[0] * model.features_float_log[:, 0]
        + (offset - theta[0]) * model.features_float_log[:, 1]
        - theta[1] * model.features_float_log[:, 2]
    )
    # Clip rate from above and below.
    clipped_dso = np.clip(log_dso, -model.max_rate, model.max_rate)
    return draw_synapses_from_poisson(model, rate=np.exp(clipped_dso))


def default_rule(
    model: RuleSimulator,
    theta: ndarray,
) -> ndarray:
    """Apply dense structural overlap rule and return synapse counts."""

    assert theta.shape == (3,), "The default DSO needs theta with shape (3,)."

    log_dso = (
        theta[0] * model.features_float_log[:, 0]
        + theta[1] * model.features_float_log[:, 1]
        - theta[2] * model.features_float_log[:, 2]
    )
    # Clip rate from above and below.
    clipped_dso = np.clip(log_dso, -model.max_rate, model.max_rate)
    return draw_synapses_from_poisson(model, rate=np.exp(clipped_dso))


def default_rule_linear(
    model: RuleSimulator,
    theta: ndarray,
) -> ndarray:
    """Apply constrained dense structural overlap rule and return synapse counts.

    The added constraint adds the **scaled** post_j target count to the denominator,
    ensuring that the denominator is never small than scaled post_j.
    """

    assert theta.shape == (3,), "The default DSO needs theta with shape (3,)."

    # Set param for fourth column containing cortical depth to zero.

    pre = theta[0] * model.features_float[:, 0]
    post = theta[1] * model.features_float[:, 1]
    postall = theta[2] * model.features_float[:, 2]
    dso = pre * post / postall

    return draw_synapses_from_poisson(model, rate=dso)


def dso_linear_two_param(
    model: RuleSimulator,
    theta: ndarray,
    offset: float = 3.0,
) -> ndarray:
    """Apply constrained dense structural overlap rule and return synapse counts.

    The added constraint adds the **scaled** post_j target count to the denominator,
    ensuring that the denominator is never small than scaled post_j.
    """

    assert theta.shape == (2,), "The default DSO needs theta with shape (2,)."

    # Set param for fourth column containing cortical depth to zero.

    pre = theta[0] * model.features_float[:, 0]
    post = (offset - theta[0]) * model.features_float[:, 1]
    postall = theta[1] * model.features_float[:, 2]
    dso = pre * post / postall

    return draw_synapses_from_poisson(model, rate=dso)


def default_rule_linear_constrained(
    model: RuleSimulator,
    theta: ndarray,
) -> ndarray:
    """Apply constrained dense structural overlap rule and return synapse counts.

    The added constraint adds the **scaled** post_j target count to the denominator,
    ensuring that the denominator is never small than scaled post_j.
    """

    assert theta.shape == (3,), "The default DSO needs theta with shape (3,)."

    # Set param for fourth column containing cortical depth to zero.

    pre = theta[0] * model.features_float[:, 0]
    post = theta[1] * model.features_float[:, 1]
    postall = theta[2] * model.postall_from_post(post)
    dso = pre * post / postall

    return draw_synapses_from_poisson(model, rate=dso)


def default_rule_constrained(
    model: RuleSimulator,
    theta: ndarray,
) -> ndarray:
    """Apply constrained dense structural overlap rule and return synapse counts.

    The added constraint adds the **scaled** post_j target count to the denominator,
    ensuring that the denominator is never small than scaled post_j.
    """

    assert theta.shape == (3,), "The default DSO needs theta with shape (3,)."

    log_pre = theta[0] * model.features_float_log[:, 0]
    log_post = theta[1] * model.features_float_log[:, 1]
    # To calculate postall from post we need to go to linear space.
    log_postall = theta[2] * np.log(model.postall_from_post(np.exp(log_post)))
    log_dso = log_pre + log_post - log_postall

    # Clip rate from above and below.
    clipped_dso = np.clip(log_dso, -model.max_rate, model.max_rate)
    return draw_synapses_from_poisson(model, rate=np.exp(clipped_dso))


def default_rule_pre_calculated_linear(
    model: RuleSimulator, pre: ndarray, theta: ndarray
):
    """Apply default rule for post and postall, given log_pre already calculated."""

    assert theta.shape == (2,), "Theta needs to only two entries: post and postall."

    # Set param for fourth column containing cortical depth to zero.

    post = theta[0] * model.features_float[:, 1]
    postall = theta[1] * model.postall_from_post(post)
    dso = pre * post / postall

    return draw_synapses_from_poisson(model, rate=dso)


def default_rule_pre_calculated(model: RuleSimulator, log_pre: ndarray, theta: ndarray):
    """Apply default rule for post and postall, given log_pre already calculated."""

    assert theta.shape == (2,), "Theta needs to only two entries: post and postall."

    # Weighted dso with single weights for post and postall.
    log_post = theta[0] * model.features_float_log[:, 1]
    log_postall = theta[1] * np.log(model.postall_from_post(np.exp(log_post)))
    log_dso = log_pre + log_post - log_postall

    # Clip rate from above and below.
    clipped_dso = np.clip(log_dso, -model.max_rate, model.max_rate)
    return draw_synapses_from_poisson(model, rate=np.exp(clipped_dso))


def rule_Layer4(
    model: RuleSimulator,
    theta: ndarray,
) -> ndarray:
    """Variante of the default rule, using separate weights for L4 boutons vs rest."""

    assert theta.shape == (
        4,
    ), "Rule needs theta with shape (4,): pre-Layer4, pre-not-Layer4, pst, pstAll"

    # Prelocate and apply separate weights on boutons.
    log_pre = np.zeros(model.features_float.shape[0], dtype=float)
    log_pre[model.is_l4] = theta[0] * model.features_float_log[model.is_l4, 0]
    log_pre[model.not_l4] = theta[1] * model.features_float_log[model.not_l4, 0]

    return default_rule_pre_calculated(model, log_pre, theta[2:])


def rule_L4ss(
    model: RuleSimulator,
    theta: ndarray,
) -> ndarray:
    """Apply separate bouton weight, if postsynaptic cell type is L4ss."""

    assert theta.shape == (
        4,
    ), "Rule needs theta with shape (4,): pre-Layer4, pre-not-Layer4, pst, pstAll"

    # Prelocate and apply separate weights on boutons.
    log_pre = np.zeros(model.features_float.shape[0], dtype=float)
    log_pre[model.is_l4ss] = theta[0] * model.features_float_log[model.is_l4ss, 0]
    log_pre[model.not_l4ss] = theta[1] * model.features_float_log[model.not_l4ss, 0]

    return default_rule_pre_calculated(model, log_pre, theta[2:])


def rule_l4_l4ss_rest(model: RuleSimulator, theta: ndarray):
    """
    Apply rule with separate boutons weights for L4ss, L4rest, and not L4.
    """

    assert theta.shape == (
        5,
    ), """Rule needs theta with shape (5,): pre-Layer4ss, pre-layer4rest, 
    pre-not-layer4, pst, pstAll"""

    # Prelocate and apply separate weights on boutons.
    log_pre = np.zeros(model.features_float.shape[0], dtype=float)
    log_pre[model.is_l4l4ss] = theta[0] * model.features_float_log[model.is_l4l4ss, 0]
    log_pre[model.is_l4rest] = theta[1] * model.features_float_log[model.is_l4rest, 0]
    # apply third weight to all remaining indices (not L4).
    log_pre[model.not_l4] = theta[2] * model.features_float_log[model.not_l4, 0]

    return default_rule_pre_calculated(model, log_pre, theta[3:])


def rule_L4ss_linear(
    model: RuleSimulator,
    theta: ndarray,
) -> ndarray:
    """Apply separate bouton weight, if postsynaptic cell type is L4ss."""

    assert theta.shape == (
        4,
    ), "Rule needs theta with shape (4,): pre-Layer4, pre-not-Layer4, pst, pstAll"

    # Prelocate and apply separate weights on boutons.
    pre = np.zeros(model.features_float.shape[0], dtype=float)
    pre[model.is_l4ss] = theta[0] * model.features_float[model.is_l4ss, 0]
    pre[model.not_l4ss] = theta[1] * model.features_float[model.not_l4ss, 0]

    return default_rule_pre_calculated_linear(model, pre=pre, theta=theta[2:])


def rule_L4ss_L4sp_L5it_l5pt_linear(
    model: RuleSimulator,
    theta: ndarray,
) -> ndarray:
    """Apply separate bouton weights for specific postsynaptic cell types L4ss."""

    assert theta.shape == (
        7,
    ), "Rule needs theta with shape (7,): l4sp, l4ss, l5it, l5pt, rest, pst, pstAll"

    # Prelocate and apply separate weights on boutons.
    pre = np.zeros(model.features_float.shape[0], dtype=float)
    pre[model.is_l4sp] = theta[0] * model.features_float[model.is_l4sp, 0]
    pre[model.is_l4ss] = theta[1] * model.features_float[model.is_l4ss, 0]
    pre[model.is_l5it] = theta[2] * model.features_float[model.is_l5it, 0]
    pre[model.is_l5pt] = theta[3] * model.features_float[model.is_l5pt, 0]
    pre[model.not_4lss_l4sp_l5it_l5pt] = (
        theta[4] * model.features_float[model.not_4lss_l4sp_l5it_l5pt, 0]
    )

    return default_rule_pre_calculated_linear(model, pre=pre, theta=theta[5:])


def rule_L4ss_L4sp_L5it_l5pt_infra_linear(
    model: RuleSimulator,
    theta: ndarray,
) -> ndarray:
    """Apply separate bouton weights for specific postsynaptic cell types and infragranular layer."""

    assert theta.shape == (
        8,
    ), "Rule needs theta with shape (8,): l4sp, l4ss, l5it, l5pt, rest_not_infra, rest_infra, pst, pstAll"

    # Prelocate and apply separate weights on boutons.
    pre = np.zeros(model.features_float.shape[0], dtype=float)
    pre[model.is_l4sp] = theta[0] * model.features_float[model.is_l4sp, 0]
    pre[model.is_l4ss] = theta[1] * model.features_float[model.is_l4ss, 0]
    pre[model.is_l5it] = theta[2] * model.features_float[model.is_l5it, 0]
    pre[model.is_l5pt] = theta[3] * model.features_float[model.is_l5pt, 0]
    pre[model.not_l4ss_l4sp_l5it_l5pt_not_infra] = (
        theta[4] * model.features_float[model.not_l4ss_l4sp_l5it_l5pt_not_infra, 0]
    )
    pre[model.not_l4ss_l4sp_l5it_l5pt_infra] = (
        theta[5] * model.features_float[model.not_l4ss_l4sp_l5it_l5pt_infra, 0]
    )

    return default_rule_pre_calculated_linear(model, pre=pre, theta=theta[6:])


def bernoulli_glm(model: RuleSimulator, theta: ndarray) -> ndarray:
    """Return Bernoulli GLM samples of present connections ijk.

    Predict the Bernoulli connection probability of neurons i and j
    in voxel k from dense structural overlap.

    Parameters theta act in log space to fulfil GLM requirements.
    """

    assert theta.shape == torch.Size([3]), "theta must have exactly 3 entries: (3,)."

    X = model.features_float_log

    pre = theta[0] * X[:, 0]
    post = theta[1] * X[:, 1]
    postall = theta[2] * X[:, 2]

    log_dso = pre + post - postall

    return np.random.binomial(n=1, p=1 / (1 + np.exp(-log_dso)))


def bernoulli_rule_linear(model: RuleSimulator, theta: ndarray) -> ndarray:
    """Return Bernoulli samples of present connections ijk.

    Predict the Bernoulli connection probability of neurons i and j
    in voxel k from dense structural overlap.

    Parameters theta act in linear space for better interpretability.
    """

    assert theta.shape == torch.Size([3]), "theta must have exactly 3 entries: (3,)."

    X = model.features_float

    pre = theta[0] * X[:, 0]
    post = theta[1] * X[:, 1]
    postall = theta[2] * X[:, 2]

    dso = pre * post / postall

    # Using the linearized version of the canonical link function.
    return np.random.binomial(n=1, p=1 / (1 + 1 / dso))


def bernoulli_rule_linear_constrained(model: RuleSimulator, theta: ndarray) -> ndarray:
    """Return Bernoulli samples of present connections ijk.

    Predict the Bernoulli connection probability of neurons i and j
    in voxel k from dense structural overlap.

    Parameters theta act in linear space for better interpretability.
    """

    assert theta.shape == torch.Size([3]), "theta must have exactly 3 entries: (3,)."

    X = model.features_float

    pre = theta[0] * X[:, 0]
    post = theta[1] * X[:, 1]
    postall = theta[2] * model.postall_from_post(post)

    dso = pre * post / postall

    # Using the linearized version of the canonical link function.
    return np.random.binomial(n=1, p=1 / (1 + 1 / dso))


def peters_rule_subcellular(model: RuleSimulator, theta: ndarray):
    # Sample with probability theta for each ijk where ij meet (feature == 1)
    # Feature is 0 wherever they do not meet.
    synapses = np.random.binomial(n=1, p=theta * model.features_float)

    return synapses.squeeze()
