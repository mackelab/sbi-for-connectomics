import numpy as np
import torch
import time

from .utils import *
from numpy import ndarray
from torch import Tensor
from typing import Callable, Tuple, Union


class PetersRuleSimulator:
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
        path_to_features,
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
                path_to_features,
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


## Stripped down version to work with subcellular variant of Peters' rule.
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
        features_float = features_float.reshape(1, -1)

        self.features_int = features_int
        self.features_float = features_float
        self.features_float_log = self.get_log_features(features_float)
        self.global_norm = self.features_float[:, 2].mean()
        self.log_global_norm = np.log(self.global_norm)
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

        self.return_synapse_counts = return_synapse_counts

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

        if isinstance(theta, Tensor):
            theta_numpy = theta.numpy()
        else:
            theta_numpy = theta

        theta_numpy = theta_numpy.squeeze()

        return theta_numpy

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
            print("loading features ...")
        features_int = np.loadtxt(filenameInt, dtype=np.int32)
        features_float = np.loadtxt(filenameFloat, dtype=np.float32)

        if self.verbose:
            print(f"Time elapsed: {time.time() - start}")

        return features_int, features_float

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


def peters_rule_subcellular(model: RuleSimulator, theta: ndarray):

    # Sample with probability theta for each ijk where ij meet (feature == 1)
    # Feature is 0 wherever they do not meet.
    synapses = np.random.binomial(n=1, p=theta * model.features_float)

    return synapses.squeeze()
