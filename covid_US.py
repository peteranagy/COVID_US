#!/usr/bin/env python
# coding: utf-8

import networkx as nx
import dynetx as dn

import ndlib.models.ModelConfig as mc
import ndlib.models.dynamic as dm
import ndlib.models.epidemics as ep

import numpy as np
import random

from tqdm import tqdm_notebook as tqdm
import itertools

# #### model parameters

# - `social_dist_range`: time frame, within which social distancing restrictions are in effect
# - `small_ent_range`: time frame, within which small enterprises are closed
# - `large_ent_range`: time frame, within which large enterprises are closed

# - `N`: number of nodes
# - `k`: average degree
# - `number_of_iterations`: observed time frame
# - `contact_duration`: all non-recurring contacts last this long
# - `normal_contacts_lambda`: mean active/all contacts ratio in one iteration (drawn from an exponential distribution)
# - `social_dist_contacts_lambda`: mean active/all contacts ratio in one iteration
# (drawn from an exponential distribution) within social distancing time frame
# - `parties_lambda`: mean attendance ratio of parties (drawn from an exponential distribution)
# - `contacts_threshold`: determines the max ratio of active contacts in one iteration
# - `number_of_families`, `family_size`: number and size of complete subgraphs present during the entire
# observed time frame
# - `small_ent_number`, `small_ent_size`: number and size of complete subgraphs present outside `small_ent_range`
# - `large_ent_number`, `large_ent_size`: number and size of complete subgraphs present outside `large_ent_range`


def find_clique_edges(cliques):

    return set(
        sum(
            [
                [frozenset(c) for c in itertools.combinations(clique, r=2)]
                for clique in cliques
            ],
            [],
        )
    )


def build_dynamic_network(
    social_dist_range,
    small_ent_range,
    large_ent_range,
    g,
    number_of_iterations=1000,
    contact_duration=50,
    normal_contacts_lambda=0.1,
    social_dist_contacts_lambda=0.001,
    parties_lambda=0.01,
    contacts_threshold=1,
    number_of_families=500,
    family_size=4,
    small_ent_number=100,
    small_ent_size=20,
    large_ent_number=20,
    large_ent_size=100,
):

    ### build static network

    N = g.number_of_nodes()
    l = g.number_of_edges()

    #### designate families
    
    np.random.seed(0)

    families = find_clique_edges(
        np.random.choice(
            g.nodes(), size=number_of_families * family_size, replace=False
        ).reshape(-1, family_size)
    )

    ### designate small and large enterprises

    enterprises = np.random.choice(
        g.nodes,
        small_ent_number * small_ent_size + large_ent_number * large_ent_size,
        replace=False,
    )

    small_ent = (
        find_clique_edges(
            enterprises[: small_ent_number * small_ent_size].reshape(-1, small_ent_size)
        )
        - families
    )

    large_ent = (
        find_clique_edges(
            enterprises[small_ent_number * small_ent_size :].reshape(-1, large_ent_size)
        )
        - families
    )

    ### find remaining edges for further contacts

    potential_edges = (
        set([frozenset(e) for e in g.edges]) - families - small_ent - large_ent
    )

    #### build dynamic network

    dg = dn.DynGraph(edge_removal=True)
    dg.add_nodes_from(g.nodes)

    #### add families and enterprises

    dg.add_interactions_from([list(f) for f in families], t=0, e=number_of_iterations)

    for r, ent in [(small_ent_range, small_ent), (large_ent_range, large_ent)]:

        if r is None:

            dg.add_interactions_from(
                [list(e) for e in ent], t=0, e=number_of_iterations
            )

        else:

            dg.add_interactions_from([list(e) for e in ent], t=0, e=r[0])
            dg.add_interactions_from(
                [list(e) for e in ent], t=r[-1], e=number_of_iterations
            )

    ### iterate over observed time frame

    for i in tqdm(range(number_of_iterations)):

        ### add basic contacts

        _lambda = (
            normal_contacts_lambda
            if ((social_dist_range is None) or (i not in social_dist_range))
            else social_dist_contacts_lambda
        )

        new_contacts = set(
            random.sample(
                potential_edges,
                int(min(np.random.exponential(_lambda), contacts_threshold) * l),
            )
        )

        ### add large gatherings

        if (social_dist_range is None) or (i not in social_dist_range):

            parties = set(
                find_clique_edges(
                    [
                        random.sample(
                            g.nodes,
                            int(
                                min(
                                    np.random.exponential(parties_lambda),
                                    contacts_threshold,
                                )
                                * N
                            ),
                        )
                    ]
                )
            ).intersection(potential_edges)

        else:
            parties = set()

        ### add all contacts

        all_contacts = [list(c) for c in new_contacts.union(parties)]

        for c in all_contacts:
            try:
                dg.add_interaction(
                    c[0], c[1], t=i, e=i + contact_duration,
                )
            except:
                pass

    return dg


# ### model parameters

### `g`: network to run simulation on
### `inf_0`: initial ratio of infected
### `beta`: transmission rate
### `gamma`: recovery rate


def SIR_model(g, inf_0=0.05, beta=0.001, gamma=0.025):

    model = dm.DynSIRModel(g)

    config = mc.Configuration()
    config.add_model_parameter("percentage_infected", inf_0)
    config.add_model_parameter("beta", beta)
    config.add_model_parameter("gamma", gamma)
    model.set_initial_status(config)

    iterations = model.execute_snapshots()
    trends = model.build_trends(iterations)

    return [model, iterations, trends]


### static SIR for benchmark


def stat_SIR_model(g, inf_0, beta, gamma):

    model = ep.SIRModel(g)

    config = mc.Configuration()
    config.add_model_parameter("fraction_infected", inf_0)
    config.add_model_parameter("beta", beta)
    config.add_model_parameter("gamma", gamma)
    model.set_initial_status(config)

    iterations = model.iteration_bunch(1000)
    trends = model.build_trends(iterations)

    return [model, iterations, trends]
