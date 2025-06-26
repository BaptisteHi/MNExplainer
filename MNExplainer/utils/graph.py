import json
import torch
import graphmuse as gm
import numpy as np
import struttura as st
import networkx as nx
import math as m
from pathlib import Path
from torch_geometric.data import HeteroData


def save_pyg_graph_as_json(graph, ids, name, extra_info=None, path="./"):
    """Save the graph as a json file.

    Args:
        graph (torch_geometric.data.HeteroData): the graph to save
    """
    out_dict = {}
    # for k,v in graph.__dict__.items():
    #     if isinstance(v, (np.ndarray,torch.Tensor)):
    #         out_dict[k] = v.tolist()
    #     elif isinstance(v, str):
    #         out_dict[k] = v
    # export the input edges
    for k, v in graph.edge_index_dict.items():
        out_dict[k[1]] = v.tolist()

    # export the output edges
    # truth edges
    # out_dict["output_edges_dict"]["truth"] = graph["truth_edges"].tolist()
    # # potential edges
    # out_dict["output_edges_dict"]["potential"] = graph["pot_edges"].tolist()

    # export the nodes ids
    if "_" in ids[0]:  # MEI with multiple parts, remove the Pxx_ prefix
        out_dict["id"] = [i.split("_")[1] for i in ids]
    # else:
    #    out_dict["id"] = ids.tolist()

    if extra_info is not None:
        for k, v in extra_info.items():
            out_dict[k] = v

    with open(Path(path, name), "w") as f:
        print("Saving graph to", Path(path, name))
        json.dump(out_dict, f)

def _to_label(cad_str):
    match cad_str:
        case "PAC":
            return 1
        case "IAC":
            return 2
        case "HC":
            return 3
        case "EC":
            return 0 #let's ignore them for now, as they are very rare
        case "DC":
            return 0 #let's ignore them for now, as they are very rare
        case "PC":
            return 0 #let's ignore them for now, as they are very rare
    return 0

def cad_label_to_string(cad):
    match cad:
        case 1:
            return 'PAC'
        case 2:
            return 'IAC'
        case 3:
            return 'HC'
        case 4:
            return 'EC'
        case 5:
            return 'DC'
        case 6:
            return 'PC'
    return ''

def id_to_int(id):
    try:
        s = id.split('p')
        key_numbers = s[1].split('n')
        return 10 * int(key_numbers[1]) + int(key_numbers[0])
    except:
        return -1

def int_to_id(n):
    n=n.item()
    if n == -1:
        return None
    else :
        part_number = int(n%10)
        note_number = int((n - part_number)//10)
        id = f'p{part_number}n{note_number}'
        return id 

def create_graph_for_score(score, pitch_encoder, include_cadence=False, include_id=False, include_ts_beats=False, include_divs_pq=False, add_beats=False):
    """
    The function for creating a heterogeneous graph out of a score object.

    Parameters
    ----------
    score : score object
        the score

    Returns
    -------
    the graph and the names of the "note" nodes features.
    """
    features, f_names = gm.utils.get_score_features(score)
    note_array = score.note_array(include_time_signature=True, include_metrical_position=True, include_pitch_spelling=True)
    # cad_features, cad_f_names = st.descriptors.utils.cadence_features.get_cad_features(score[-1], note_array)
    # complete_features = np.concatenate((features, cad_features), axis=1)
    complete_features = features    
    complete_features_names = f_names # + cad_f_names
    graph = gm.create_score_graph(complete_features, score.note_array(), add_beats=add_beats)
    score_cadences = score[-1].cadences
    notes_onsets = graph["note"].onset_div
    n = len(notes_onsets) #number of nodes in the graph / notes in the score
    note_level_cadence = torch.tensor([0 for i in range(n)]) 
    for i in range(n):
        for cad in score_cadences:
            if notes_onsets[i] == cad.start.t:
                note_level_cadence[i] = _to_label(cad.text)
    if include_cadence:
        graph["note"].cadences = note_level_cadence
    if include_id:
        graph["note"].id = note_array['id']
        # graphs created using this bug the dataloader process for the training because of the strings
    if include_ts_beats:
        graph['note'].ts_beats = note_array['ts_beats']
    if include_divs_pq:    
        graph['note'].divs_pq = note_array['divs_pq']
    graph["note"].pitch_spelling = torch.tensor(pitch_encoder.encode(score[0].note_array(include_pitch_spelling=True))).long()
    return graph, complete_features_names

def hgraph_to_networkx(graph : HeteroData, computation_notes, edge_tuple):
    """unused for now and very little optimized"""
    graph_edges = graph.edge_index_dict[edge_tuple]
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(computation_notes)
    for i in range(len(graph_edges[0])):
        u,v = graph_edges[0,i], graph_edges[1,i]
        if u in computation_notes and v in computation_notes:
            nx_graph.add_edge(u,v)
    return nx_graph
