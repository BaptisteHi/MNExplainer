import torch

def counterfactual_score(model, target, input_graph, desired_label, ce_graph, changes, computation_notes, **kwargs):
    """
    A function computing a score for a counterfactual explanation. Its aim is to measure 'how good' is a CE at being
    close to the model input data which prediction is being explained, changing the prediction of the model and more
    generally at giving insights of how the model works. This is all in the context of music analysis classification GNNs.

    Parameters
    ----------
    model : torch.nn.Module
        The model being explained.
    input_graph : HeteroData
        The input data which prediction is being explained by the counterfactual being rated.
    ce_graph : HeteroData
        The counterfactual explanation.

    Returns
    -------
    score : int
        the score of the counterfactual explanation.
    """
    score = {'distance' : 0, 'counterfactual' : 0, 'insight' : []}

    # Checking if the explanation flips the label to the desired one, which is its counterfactual aspect (except if the 
    # desired label has been set to the already predicted label).

    if isinstance(target, int):
        ce_pred_label = model(ce_graph.x_dict, ce_graph.edge_index_dict, **kwargs).argmax(dim=1)[target]
        score['counterfactual'] = int(ce_pred_label == desired_label)
    elif isinstance(target, tuple):
        target_edge = None
        # finding the edge in the edge dict
        u = target[0]
        v = target[1]
        for _, tensor in ce_graph.edge_index_dict.items():
            pot_edges = torch.where((tensor[0,:] == u) & (tensor[1,:] == v))[0]
            if len(pot_edges) > 0:
                target_edge = pot_edges[0]
                break
        if target_edge is None:
            assert False, 'could not find the target edge in the graph'
        ce_pred_label = model(ce_graph.x_dict, ce_graph.edge_index_dict, **kwargs).argmax(dim=1)[target_edge]
        score['counterfactual'] = int(ce_pred_label == desired_label)
    else:
        ce_pred_label = model(ce_graph.x_dict, ce_graph.edge_index_dict, **kwargs).argmax(dim=1)
        score['counterfactual'] = int(ce_pred_label == desired_label)
    
    # Checking the insights provided by the explanation

    for chg in changes:
        op = chg.operation
        match op:
            case 'pitch':
                pass
            case 'duration':
                pass
            case 'pitch_spelling':
                pass
            case 'octave':
                pass
            case 'add':
                pass
            case 'remove':
                pass
            case 'onset':
                pass
            case _:
                assert False, "One of the operations performed is not recognized by the counter factual score computer"
    
    # Computing the distance. The computation is the same used for the distance term in the loss, the term ensuring minimal distance.

    dnodes = 0
    dgraph = 0
    computation_notes.sort()
    computation_notes_mask_i = torch.tensor([i in computation_notes for i in range(len(input_graph.x_dict['note']))])
    computation_notes_mask_n = torch.tensor([i in computation_notes for i in range(len(ce_graph.x_dict['note']))])

    # computing Manhattan distance between the input and the explanation computation subgraphs
    dnodes = torch.sum(torch.abs(input_graph.x_dict['note'][computation_notes_mask_i] - ce_graph.x_dict['note'][computation_notes_mask_n]))
        
    # computing the graph edit distance between the computation subgraphs
    removed_edges = []
    added_edges = []
    added_notes_ind = []
    removed_notes_ind = []

    for chg in changes[1:]:
        if chg.operation == 'add':
            add_idx = chg.note_index
            added_notes_ind.append(add_idx)
            # lookig if the added note has exactly the properties of a note that got removed in previous changes.
            # for r_note in removed_notes_ind:
                # if graph.x_dict['note'][add_idx] == graph.x_dict['note'][r_note]:
                    # pass
            dnodes += torch.sum(ce_graph.x_dict['note'][add_idx])
        if chg.operation == 'remove':
            rmv_idx = chg.note_index
            removed_notes_ind.append(rmv_idx)
            if rmv_idx in added_notes_ind:
                # the note removed was added before, hence there is no difference in the input graph and the current graph regarding this note.
                # we remove the distance that we added when the note was added in a previous change.
                # Note : this could happend the other way around : a note sharing the same properties as a removed note is added. The 
                # problem with the other way around is to also balance the distance induced by the edges in dgraph due to the removal
                # of the note because these edges will not exactly be re-added in the final graph with the new note sharing the same
                # properties, its index being brand new. Therefore, this scenario remains treated as if the new note was a different note as the
                # one removed (which, tecnically, is accurate).
                dnodes -= torch.sum(ce_graph.x_dict['note'][rmv_idx])
            else:
                dnodes += torch.sum(ce_graph.x_dict['note'][rmv_idx])
            
        rem = chg.removed_edges
        add = chg.added_edges
        removed_edges += rem
        added_edges += add
        
    set_rem = set(removed_edges)
    set_add = set(added_edges)
    removed_edges_unique = list(set_rem)
    added_edges_unique = list(set_add) 

    added_and_removed_edges_exact = list(set_rem & set_add) 
    # Contains all the edges of a certain type that were added and removed

    rem_without_type = [e[0:2] for e in removed_edges]
    add_without_type = [e[0:2] for e in added_edges]
    set_rem_notype = set(rem_without_type)
    set_add_notype = set(add_without_type)

    added_and_removed_including_substitutions = list(set_rem_notype & set_add_notype) 
    # Contains all the edges that were removed or added regardless of their type, which includes the edges
    # that simply require a type substitution.

    n_substitutions = len(added_and_removed_including_substitutions) - len (added_and_removed_edges_exact)
        
    dgraph = max(0, len(removed_edges_unique) + len (added_edges_unique) - n_substitutions - 2*len(added_and_removed_edges_exact))

    score['distance'] = dnodes + dgraph
    
    return score