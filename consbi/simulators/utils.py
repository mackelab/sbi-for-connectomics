import itertools
import json
import os
import random

from functools import cmp_to_key

import numpy as np
import torch


def getInferenceExperimentSpec(experiments, identifier):
    for experiment in experiments:
        if experiment["identifier"] == identifier:
            return experiment
    raise RuntimeError("Unknown experiment: {}".format(identifier))


def loadJsonData(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


def loadPublications(filename, extend_sort=False):
    with open(filename) as f:
        publications = json.load(f)
    if extend_sort:
        for publication in publications:
            selectionDescriptor = getSelectionDescriptor(publication)
            publication["selection_descriptor"] = selectionDescriptor
            publication["first_author_year"] = getFirstAuthorYearDescriptor(publication)
        sortPublications(publications)
    return publications


def sortPublications(publications):
    publications.sort(key=cmp_to_key(cmpFunction))


def getSelectionDescriptor(publication):
    measurementType = publication["type"]
    if measurementType == "connection_probability_invivo":
        invivo_invitro = "invivo"
    elif measurementType == "connection_probability_invitro":
        invivo_invitro = "invitro"
    else:
        raise RuntimeError("unknown measurement type {}".format(measurementType))
    descriptor = "{}_{}-{}-->{}-{}".format(
        invivo_invitro,
        publication["pre_layer"],
        publication["pre_type"],
        publication["post_layer"],
        publication["post_type"],
    )
    if publication["only_septum"] == "yes":
        descriptor += "-septum"
    return descriptor


def getFirstAuthorYearDescriptor(publication):
    author = getFirstAuthorLastName(publication["authors"])
    return "{} et al. {}".format(author, publication["year"])


def getFirstAuthorLastName(authors):
    firstAuthor = authors.split(",")[0]
    return firstAuthor.split(" ")[-1]


def cmpLayer(a, b):
    if a == "VPM" and b != "VPM":
        return -1
    elif a != "VPM" and b == "VPM":
        return 1
    elif a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0


def cmpCellType(a, b):
    cellTypes = getCellTypes()
    cellTypes.insert(0, "EXC")
    if cellTypes.index(a) < cellTypes.index(b):
        return -1
    elif cellTypes.index(a) > cellTypes.index(b):
        return 1
    else:
        return 0


def cmpScalar(a, b):
    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0


def cmpFunction(a, b):
    preLayer = cmpLayer(a["pre_layer"], b["pre_layer"])
    postLayer = cmpLayer(a["post_layer"], b["post_layer"])
    preCellType = cmpCellType(a["pre_type"], b["pre_type"])
    postCellType = cmpCellType(a["post_type"], b["post_type"])
    year = cmpScalar(a["year"], b["year"])
    probability = cmpScalar(a["cp_empirical"], b["cp_empirical"])

    if a["ID"] == b["ID"]:
        return 0
    elif preLayer != 0:
        return preLayer
    elif preCellType != 0:
        return preCellType
    elif postLayer != 0:
        return postLayer
    elif postCellType != 0:
        return postCellType
    elif year != 0:
        return year
    elif probability != 0:
        return probability
    else:
        return 0


def getCellTypes():
    return [
        "L2PY",
        "L3PY",
        "L4PY",
        "L4sp",
        "L4ss",
        "L5IT",
        "L5PT",
        "L6ACC",
        "L6BCC",
        "L6CT",
        "INH",
        "VPM",
    ]


def filterByIds(publications, ids):
    if not len(ids):
        return publications
    publicationsFiltered = []
    for publication in publications:
        if publication["ID"] in ids:
            publicationsFiltered.append(publication)
    return publicationsFiltered


def getConstraints(publications, featureSetName, maskFolder):
    masks = []
    observables = []
    descriptors = []
    authors = []
    ids = []
    for publication in publications:
        maskFile = os.path.join(
            maskFolder, "mask_{}_{}.txt".format(featureSetName, publication["ID"])
        )
        mask = np.loadtxt(maskFile, dtype="uint32")
        masks.append(mask)
        observables.append(0.01 * publication["cp_empirical"])
        descriptors.append(publication["selection_descriptor"])
        authors.append(publication["first_author_year"])
        ids.append(publication["ID"])
    constraints = {
        "masks": masks,
        "observables": observables,
        "selectionDescriptors": descriptors,
        "authors": authors,
        "ids": ids,
    }
    return constraints


def pruneFeatureSetSubcellular(featureSetInt, featureSetFloat, masks):
    allMasks = np.concatenate(masks)
    rows = np.unique(allMasks)
    prunedFeatureSetInt = featureSetInt[rows, :]
    prunedFeatureSetFloat = featureSetFloat[rows, :]
    masksNew = []
    for mask in masks:
        newMask = np.nonzero(np.isin(rows, mask))[0]
        masksNew.append(newMask)
    return prunedFeatureSetInt, prunedFeatureSetFloat, masksNew


def binSynapseCounts(ijk_synapse_counts, ijk_depth_idx_mapping, num_bins):
    """Returns average number of synapse per cortical depth.

    NOTE: Slow version with for loops.

    Given synapse counts from the rule simulator for each ijk combination,
    and a mapping from ijk to the corresponding cortical depth,
    add up the synpase counts for each cortical depth index.
    """

    assert (
        ijk_synapse_counts.shape[0] == ijk_depth_idx_mapping.shape[0]
    ), "counts and idx mapping must match."

    synapses_per_depth = np.zeros(num_bins)
    for i in range(0, ijk_synapse_counts.shape[0]):
        bin_idx = ijk_depth_idx_mapping[i]
        if bin_idx >= 0:
            synapses_per_depth[bin_idx] += ijk_synapse_counts[i]
    return synapses_per_depth


def collect_synapse_counts_over_depth(
    ijk_synapse_counts, ijk_depth_idx_mapping, num_bins
):
    """Returns average number of synapse per cortical depth.

    NOTE: Fast version with for logical indexing.

    Given synapse counts from the rule simulator for each ijk combination,
    and a mapping from ijk to the corresponding cortical depth,
    add up the synpase counts for each cortical depth index.
    """
    # find indices that belong to the VPM pairs AND have a synapse
    idx_with_synapse = np.where(
        np.logical_and(ijk_depth_idx_mapping >= 0, ijk_synapse_counts > 0)
    )[0]

    # count how often this occurs for each cortical depths
    depth_idx, counts = np.unique(
        ijk_depth_idx_mapping[idx_with_synapse], return_counts=True
    )

    synapses_per_depth = np.zeros(num_bins)
    synapses_per_depth[depth_idx] = counts

    return synapses_per_depth


def seed_all_backends(seed: int = None) -> None:
    """Sets all python, numpy and pytorch seeds."""

    if seed is None:
        seed = int(torch.randint(1_000_000, size=(1,)))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore