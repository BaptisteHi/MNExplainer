import torch
import random
import numpy as np
import copy
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.data import HeteroData
from graphmuse.utils.graph_utils import trim_to_layer
from math import tanh

from tqdm import tqdm

from typing import List, Union, Tuple

class Change_:
    """
    A class to summarize the information relevant to one step of the noising process : what operation was performed, on what note if it 
    was a note change, etc.

    Parameters:
    -----------
    operation : string
        the operation performed, either 'pitch', 'onset', 'duration', 'add' or 'remove' for now.
    note_index (optional) : int
        the graph index of the note being changed if the operation affects some existing note, such as pitch change.
    pitch (optional) : int
        the new pitch for a pitch change or the pitch of a added note if operation is 'add'.
    octave (optional) : int
        the octave of a added note if operation is 'add'.
    onset (optional) : int
        the new onset for a onset change or the onset of a added note if operation is 'add'.
    duration (optional) : int
        the new duration for a duration change or the duration of a added note if operation is 'add'.
    """
    def __init__(self, operation, note_index=0, pitch=0, octave=0, onset=0, duration=0):
        self.operation = operation
        self.note_index = note_index
        self.pitch = pitch
        self.octave = octave
        self.onset = onset
        self.duration = duration
    
    def __str__(self):
        match self.operation:
            case 'add':
                return f'Adding a new note with pitch {self.pitch} and octave {self.octave} at the onset {self.onset} with a duration {self.duration}'
            case 'remove':
                return f'Isolating the node which index is {self.note_index}'
            case 'onset': 
                return f'Ajusting the onset of the note which corresponding node index is {self.note_index} to the onset {self.onset}'
            case 'duration':
                return f'Ajusting the duration of the note which corresponding node index is {self.note_index} to the duration {self.duration}'
            case 'pitch':
                return f'Ajusting the pitch of the note which corresponding node index is {self.note_index} to the pitch {self.pitch}'
            case _:
                assert False, 'The operation of this change object is not recognized or cannot be printed'

class MNExplainer(ExplainerAlgorithm):
    """
    The musical noise explainer class for providing counter factual explanations for a GNN musical model's predictions.

    Parameters
    ----------
    metadata : tuple or list
        a two elements list or a couple of the node types and the edge tuples of the heterogeneous data handled by the model.
    num_feat : int
        the number of "note" type nodes features.
    pred_level : string
        the prediction level of the model : either "node", "edge", or "graph".
    num_layers (optional) : int
        the number of convolution layers for the encoders in the explainer.
    lr (optional) : float
        the learning rate of the model handling the noising process.
    balance_factor (optional) : float 
        the factor balancing the importance between the minimality and the counterfactual aspect of the prediction in the loss function.
    """

    def __init__(self, model, metadata, num_feat, pred_level, num_layers=3, epochs=50, lr=0.1, balance_factor = 1.):
        super().__init__()

        self.metadata = metadata
        self.num_feat = num_feat
        self.pred_level = pred_level
        self.num_layers = num_layers
        self.epochs = epochs
        self.model = model
        self.lr = lr
        self.balance_factor = balance_factor

        self.mnmodel = MNModel_(metadata, num_feat, num_layers)

        torch.manual_seed(0)

    def forward(self, graph, desired_classification, target:Union[int, Tuple]=None, num_expl:Union[int, List]=1, retrieve_changes=False, **kwargs): 
        base_graph = copy.deepcopy(graph)
        explanations = [base_graph]
        device = base_graph.x_dict['note'].device
        self.mnmodel.to(device)
        num_notes = len(base_graph.x_dict['note'])
        not_removed_notes = [True for i in range(num_notes)]# torch.tensor([True for i in range(len(graph['note'].x))], device=device)
        self.model.eval()
        
        # Using gradient to find notes involved in the target prediction. These notes are the candidates for the noisy notes in the explanations
        # since they are the ones that impact the prediction of the mode being explained.
        if target is not None:
            computation_notes = []
            base_graph['note'].x.requires_grad = True
            pred = self.model(base_graph.x_dict, base_graph.edge_index_dict, **kwargs)
            toy_loss = torch.nn.CrossEntropyLoss()
            
            if isinstance(target, int):
                target_mask = torch.tensor([i == target for i in range(len(base_graph['note'].x))], device=device)
                toy_ground_truth = torch.tensor([0 for i in range(len(base_graph['note'].x))], device=device)
                loss = toy_loss(pred[target_mask], toy_ground_truth[target_mask])
                loss.backward()
                grad = base_graph['note'].x.grad
                for note in range(len(grad)):
                    for feature_g in grad[note]:
                        if feature_g != 0.0:
                            computation_notes.append(note)
                        break # to avoid adding the same note multiple times.

            elif isinstance(target, tuple):
                target_mask = torch.tensor([(i == target[0]) or (i == target[1]) for i in range(len(base_graph['note'].x))])
                toy_ground_truth = torch.tensor([0 for i in range(base_graph.num_edges)])
                loss = toy_loss(pred[target_mask], toy_ground_truth[target_mask])
                loss.backward()

            else :
                toy_ground_truth = torch.tensor([0])
                loss = toy_loss(pred, toy_ground_truth)
                loss.backward()

            base_graph['note'].x.requires_grad = False

        changes = [None]

        if isinstance(num_expl, int):
            for _ in tqdm(range(1,num_expl + 1)):
                self._train(self.mnmodel, self.model, base_graph, computation_notes, desired_classification, changes, target=target, **kwargs)
                noisy_x_dict, noisy_edge_index_dict, noisy_ts_beats, noisy_onset_div, noisy_duration_div, noise = self.mnmodel(
                    base_graph.x_dict,
                    base_graph.edge_index_dict,
                    base_graph['note'].ts_beats,
                    base_graph['note'].divs_pq,
                    base_graph['note'].onset_div,
                    base_graph['note'].duration_div,
                    not_removed_notes,
                    computation_notes,
                    target=target,
                )
                
                changes.append(noise)

                noisy_graph = HeteroData()
                for tuple, tensor in noisy_edge_index_dict.items():
                    noisy_graph[tuple].edge_index = tensor
                for node_type, tensor in noisy_x_dict.items():
                    noisy_graph[node_type].x = tensor
                noisy_graph['note'].onset_div = noisy_onset_div
                noisy_graph['note'].duration_div = noisy_duration_div
                noisy_graph['note'].ts_beats = noisy_ts_beats
                noisy_graph['note'].divs_pq = base_graph['note'].divs_pq
                # noisy_graph.x_dict = noisy_x_dict
                # noisy_graph.edge_index_dict = noisy_edge_index_dict
                noisy_graph.to(device)
                explanations.append(noisy_graph)
                base_graph = noisy_graph
                # print('counter prediction :')
                # print(self.model(noisy_x_dict, noisy_edge_index_dict, **kwargs).argmax(dim=1)[target])
                # print('original graph prediction :')
                # print(self.model(explanations[0].x_dict, explanations[0].edge_index_dict, **kwargs).argmax(dim=1)[target])
        else:
            for _, operation in tqdm(enumerate(num_expl)):
                self._train(self.mnmodel, self.model, base_graph, computation_notes, desired_classification, changes, target=target, operation=operation, **kwargs)
                noisy_x_dict, noisy_edge_index_dict, noisy_ts_beats, noisy_onset_div, noisy_duration_div, noise = self.mnmodel(
                    base_graph.x_dict,
                    base_graph.edge_index_dict,
                    base_graph['note'].ts_beats,
                    base_graph['note'].divs_pq,
                    base_graph['note'].onset_div,
                    base_graph['note'].duration_div,
                    not_removed_notes,
                    computation_notes,
                    operation=operation,
                    target=target
                )
                
                changes.append(noise)

                noisy_graph = HeteroData()
                for tuple, tensor in noisy_edge_index_dict.items():
                    noisy_graph[tuple].edge_index = tensor
                for node_type, tensor in noisy_x_dict.items():
                    noisy_graph[node_type].x = tensor
                noisy_graph['note'].onset_div = noisy_onset_div
                noisy_graph['note'].duration_div = noisy_duration_div
                noisy_graph['note'].ts_beats = noisy_ts_beats
                noisy_graph['note'].divs_pq = base_graph['note'].divs_pq
                # noisy_graph.x_dict = noisy_x_dict
                # noisy_graph.edge_index_dict = noisy_edge_index_dict
                noisy_graph.to(device)
                explanations.append(noisy_graph)
                base_graph = noisy_graph
                # print('counter prediction :')
                # print(self.model(noisy_x_dict, noisy_edge_index_dict, **kwargs).argmax(dim=1)[target])
                # print('original graph prediction :')
                # print(self.model(explanations[0].x_dict, explanations[0].edge_index_dict, **kwargs).argmax(dim=1)[target])

        if retrieve_changes:
            return explanations, changes
        
        return explanations
    
    def _train(self, mnmodel, model, graph, computation_notes, desired_classification, changes, target=None, operation='model_choice', **kwargs):

        optimizer = torch.optim.Adam(mnmodel.parameters(), lr=self.lr, weight_decay=0.0005)
        device = graph.x_dict['note'].device
        not_removed_notes = [True for i in range(len(graph['note'].x))]# torch.tensor([True for i in range(len(graph['note'].x))], device=device)
        for _ in tqdm(range(1, self.epochs + 1)):
            mnmodel.train()
            optimizer.zero_grad()
            noisy_x_dict, noisy_edge_index_dict, _, _, _, change = self.mnmodel(
                    graph.x_dict,
                    graph.edge_index_dict,
                    graph['note'].ts_beats,
                    graph['note'].divs_pq,
                    graph['note'].onset_div,
                    graph['note'].duration_div,
                    not_removed_notes,
                    computation_notes,
                    operation=operation,
                    target=target,
                    in_training=True
                )
            """
            if isinstance(target, int):
                pred = model(noisy_x_dict, noisy_edge_index_dict, **kwargs)[target]
                original_pred = model(graph.x_dict, graph.edge_index_dict, **kwargs)[target]
            elif isinstance(target, tuple):
                target_edge = None
                # finding the edge in the edge dict
                u = target[0]
                v = target[1]
                for _, tensor in noisy_edge_index_dict.items():
                    pot_edges = torch.where((tensor[0,:] == u) & (tensor[1,:] == v))[0]
                    if len(pot_edges) > 0:
                        target_edge = pot_edges[0]
                        break
                if target_edge is None:
                    assert False, 'could not find the target edge in the graph'
                pred = model(noisy_x_dict, noisy_edge_index_dict, **kwargs)[target_edge]
                original_pred = model(graph.x_dict, graph.edge_index_dict, **kwargs)[target_edge]
            elif target is None:
                pred = model(noisy_x_dict, noisy_edge_index_dict, **kwargs)
                original_pred = model(graph.x_dict, graph.edge_index_dict, **kwargs)
            else:
                assert False, 'the target is not recognized as a node index, nor an edge tuple, nor is None for graph classification.'
            """
            pred = model(noisy_x_dict, noisy_edge_index_dict, **kwargs)
            original_pred = model(graph.x_dict, graph.edge_index_dict, **kwargs)
            loss = self._loss(change, changes, pred, original_pred, graph, target, desired_classification, self.balance_factor)
            # print(f'Loss : {loss}')
            loss.backward()
            optimizer.step()

    def _dist_from_change_naive(self, change : Change_, graph):
        # naive distance : each change of a note property is 2 because it could be made by removing and re-adding a modified version of a note
        match change.operation:
            case 'add':
                return 0.1
            case 'remove':
                return 0.1
            case 'onset':
                return 0.2
            case 'duration':
                return 0.2
            case 'pitch':
                return 0.2
        return 0
    
    def _loss(self, change, changes, pred, original_pred, graph, target, desired_classification, balance_factor):

        ent = torch.nn.CrossEntropyLoss()
        desired_pred = original_pred.argmax(dim=1)
        desired_pred[target] = desired_classification
        # we apply a mask so only the prediction for the target is taken into account
        n_notes = len(desired_pred)
        if change.operation == 'add':
            n_notes += 1
            desired_pred = torch.cat((desired_pred, torch.tensor([0], device=desired_pred.device)))
            # we put 0 as a desired classification for the new note because we won't look at it anyways
        target_mask = torch.tensor([i == target for i in range(n_notes)])

        d = self._dist_from_change_naive(change, graph)

        for chg in changes:
            if chg != None:
                d += self._dist_from_change_naive(chg, graph)

        # print(f'classification sans changement{original_pred.argmax(dim=1)[target]}')
        # print(f'classification avec changement{pred.argmax(dim=1)[target]}')

        return balance_factor * ent(pred[target_mask], desired_pred[target_mask]) + d

    
    def supports(self):
        return True
    
class MNModel_(torch.nn.Module):
    """
    The class for musical noise explainer noising model.

    Parameters
    ----------
    metadata : tuple or list
        a two elements list or a couple of the node types and the edge tuples of the heterogeneous data handled by the model.
    num_feat : int
        the number of "note" type nodes features.
    num_layers : int
        the number of convolution layers for the encoders in the explainer.   
    """

    def __init__(self, metadata, num_feat, num_layers):
        super().__init__()

        self.metadata = metadata
        self.num_feat = num_feat
        self.num_layers = num_layers

        """
        Encoder for choosing the operation to perform when unspecified. The loss depends on the nature of the operation because of the
        counterfactual explanation distance term, hence the needs for an encoder that can me tuned.
        """
        self.operation_choice = EncodingGNN_(metadata, num_feat, 5, num_layers)

        """Encoders for the remove operator"""
        self.removing_note_modules = torch.nn.ModuleDict({
            'index' : EncodingGNN_(metadata, num_feat, 2, num_layers)
        })

        """Encoders for the pitch update operator"""
        self.pitch_change_modules = torch.nn.ModuleDict({
            'index' : EncodingGNN_(metadata, num_feat, 2, num_layers),
            'pitch' : EncodingGNN_(metadata, num_feat, 12, num_layers)
        })

        """Encoders for the adding note operator"""
        self.add_note_modules = torch.nn.ModuleDict({
            'index' : EncodingGNN_(metadata, num_feat, 2, num_layers),
            'pitch' : EncodingGNN_(metadata, num_feat, 12, num_layers),
            'octave' : EncodingGNN_(metadata, num_feat, 10, num_layers),
            'onset' : EncodingGNN_(metadata, num_feat, 2, num_layers),
            'duration' : EncodingGNN_(metadata, num_feat, 4, num_layers)
            })

        """Encoders for the onset update operator"""
        self.onset_change_modules = torch.nn.ModuleDict({
            'index' : EncodingGNN_(metadata, num_feat, 2, num_layers),
            'onset' : EncodingGNN_(metadata, num_feat, 2, num_layers)
        })

        """Encoders for the duration update operator"""
        self.duration_change_modules = torch.nn.ModuleDict({
            'index' : EncodingGNN_(metadata, num_feat, 2, num_layers),
            'duration' : EncodingGNN_(metadata, num_feat, 4, num_layers)
        })
    
    def forward(self,
                x_dict,
                edge_index_dict,
                ts_beats,
                divs_pq,
                onset_div,
                duration_div,
                not_removed_notes,
                computation_notes,
                operation='model_choice',
                target=None,
                in_training=False):

        if operation=='model_choice':
            possible_operations = ['pitch','onset','duration','add','remove']
            # op_idx = random.randint(0,len(possible_operations)-1)
            embeddings_for_operation_choice = self.operation_choice(x_dict, edge_index_dict)
            op_idx = torch.argmax(embeddings_for_operation_choice['note'][target])
            operation = possible_operations[op_idx]

        computation_notes.sort()
        computation_notes_mask = torch.tensor([i in computation_notes for i in range(len(x_dict['note']))])

        match operation:
            case 'pitch':
                # We select a note in the computation nodes to adjust its pitch.
                embeddings_for_note_index = self.pitch_change_modules['index'](x_dict, edge_index_dict)
                embeddings_for_new_pitch = self.pitch_change_modules['pitch'](x_dict, edge_index_dict)
                computation_notes_scores = torch.sum(embeddings_for_note_index['note'], dim=1)[computation_notes_mask]
                note_index = computation_notes[torch.argmax(computation_notes_scores)]
                # note_index.to(x_dict['note'].device)
                new_pitch = torch.argmax(embeddings_for_new_pitch['note'][note_index])
                new_pitch.to(x_dict['note'].device)

                new_x_dict = self._update_pitch(new_pitch, note_index, x_dict)
                new_edge_index_dict = dict(edge_index_dict)

                update = Change_('pitch', note_index=note_index, pitch=new_pitch)

                return new_x_dict, new_edge_index_dict, ts_beats, onset_div, duration_div, update
            
            case 'onset':
                # We select two notes from the computing nodes : the first one will have its onset adjusted to the onset of the second one.
                embeddings_for_note_index = self.onset_change_modules['index'](x_dict, edge_index_dict)
                computation_notes_scores = torch.sum(embeddings_for_note_index['note'], dim=1)[computation_notes_mask]
                note_index = computation_notes[torch.argmax(computation_notes_scores)]
                embeddings_for_new_onset = self.onset_change_modules['onset'](x_dict, edge_index_dict)
                computation_notes_scores_onset = torch.sum(embeddings_for_new_onset['note'], dim=1)[computation_notes_mask]
                new_onset_note = computation_notes[torch.argmax(computation_notes_scores_onset)]
                new_onset = onset_div[new_onset_note]
            
                new_edge_index_dict, noisy_onset_div = self._update_onset(
                    new_onset, note_index, onset_div, duration_div, edge_index_dict, not_removed_notes)
                new_x_dict = dict(x_dict)
                
                update = Change_('onset', note_index=note_index, onset=new_onset)

                return new_x_dict, new_edge_index_dict, ts_beats, noisy_onset_div, duration_div, update

            case 'duration':
                # We choose a note in the computation nodes that will have its duration modified. The new duration is chosen among full, half,
                # quarter or eighth notes.
                embeddings_for_note_index = self.duration_change_modules['index'](x_dict, edge_index_dict)
                computation_notes_scores = torch.sum(embeddings_for_note_index['note'], dim=1)[computation_notes_mask]
                note_index = computation_notes[torch.argmax(computation_notes_scores)]
                embeddings_for_new_duration = self.duration_change_modules['duration'](x_dict, edge_index_dict)
                new_duration = (1 + torch.argmax(embeddings_for_new_duration['note'][note_index])) * divs_pq[0]/2
                # look at how are the duration computed to find the multiplier.

                new_x_dict, new_edge_index_dict, noisy_duration_div = self._update_duration(
                    new_duration, note_index, ts_beats, onset_div, duration_div, x_dict, edge_index_dict, not_removed_notes)
                
                update = Change_('duration', note_index=note_index, duration=new_duration)

                return new_x_dict, new_edge_index_dict, ts_beats, onset_div, noisy_duration_div, update
            
            case 'add':
                embeddings_for_new_note_onset = self.add_note_modules['onset'](x_dict, edge_index_dict)
                embeddings_for_new_note_duration = self.add_note_modules['duration'](x_dict, edge_index_dict)
                embeddings_for_new_note_pitch = self.add_note_modules['pitch'](x_dict, edge_index_dict)
                embeddings_for_new_note_octave = self.add_note_modules['octave'](x_dict, edge_index_dict)
                embeddings_for_note_index = self.add_note_modules['index'](x_dict, edge_index_dict)

                if target is None:
                    note_index = torch.argmax(embeddings_for_note_index['note'])
                    new_note_pitch = torch.argmax(embeddings_for_new_note_pitch['note'][note_index])
                    new_note_octave = torch.argmax(embeddings_for_new_note_octave['note'][note_index])
                    new_note_onset = torch.tensor(round(float(embeddings_for_new_note_onset['note'][note_index])), device=x_dict['note'].device)
                    new_note_duration = torch.tensor(round(float(embeddings_for_new_note_duration['note'][note_index])), device=x_dict['note'].device)
                elif isinstance(target, int):
                    note_index = target
                    new_note_pitch = torch.argmax(embeddings_for_new_note_pitch['note'][note_index])
                    new_note_octave = torch.argmax(embeddings_for_new_note_octave['note'][note_index])
                    computation_notes_scores_onset = torch.sum(embeddings_for_new_note_onset['note'], dim=1)[computation_notes_mask]
                    onset_note = computation_notes[torch.argmax(computation_notes_scores_onset)]
                    new_note_onset = torch.tensor(onset_div[onset_note], device='cpu') #can cause some warnings if some operations occured before because onset_div then becomes a tensor.
                    new_note_duration = int((1 + torch.argmax(embeddings_for_new_note_duration['note'][note_index])) * divs_pq[0]/2)

                elif isinstance(target, tuple):
                    note_index1 = target[0]
                    note_index2 = target[1]
                    new_note_pitch = torch.argmax(
                        torch.add(embeddings_for_new_note_pitch['note'][note_index1], embeddings_for_new_note_pitch['note'][note_index2]))
                    new_note_octave = torch.argmax(
                        torch.add(embeddings_for_new_note_octave['note'][note_index1], embeddings_for_new_note_octave['note'][note_index2]))
                    new_note_onset = round(0.5*
                        (float(embeddings_for_new_note_onset['note'][note_index1])+float(embeddings_for_new_note_onset['note'][note_index2])))
                    new_note_duration = round(0.5*
                        (float(embeddings_for_new_note_duration['note'][note_index1])+float(embeddings_for_new_note_onset['note'][note_index2])))
                else:
                    assert False, 'The target provided is not recognized'

                feature_vector = torch.zeros(1, self.num_feat, device=x_dict['note'].device, requires_grad=False)
                feature_vector[0,1+new_note_pitch] = 1.0
                feature_vector[0,13+new_note_octave] = 1.0
                feature_vector[0,0] = 1 - tanh(new_note_duration/ts_beats[0])
                not_removed_notes.append(True)
                # note : we assumed here that the time signature beats is always the same in the graph. If it is not the case, we have
                # to find a note (maybe near the new note onset) that would in theory share the time signature beats ?
                # in the worse case, we can also just keep a random time signature beats and let the model adapt the duration
                # prediction to make sense.

                new_x_dict, new_edge_index_dict, noisy_onset_div, noisy_duration_div = self._add_note(
                    x_dict, edge_index_dict, new_note_onset, new_note_duration, onset_div, duration_div, feature_vector, not_removed_notes)
                # noisy_ts_beats = np.concatenate((ts_beats,np.array([0])),axis=None)
                try:
                    ts_beats.to(x_dict['note'].device)
                except:
                    ts_beats = torch.tensor(ts_beats, device=x_dict['note'].device)
                noisy_ts_beats = torch.cat((ts_beats, torch.tensor([0],device=x_dict['note'].device)))

                update = Change_('add', pitch=new_note_pitch, octave=new_note_octave, onset=new_note_onset, duration=new_note_duration)

                return new_x_dict, new_edge_index_dict, noisy_ts_beats, noisy_onset_div, noisy_duration_div, update

            case 'remove':
                # We select a note from the computation nodes and we isolate it in the score graph so it is no longer relevant for the message
                # passing operations.
                embeddings_for_note_index = self.removing_note_modules['index'](x_dict, edge_index_dict)
                computation_notes_scores = torch.sum(embeddings_for_note_index['note'], dim=1)[computation_notes_mask]
                note_index = computation_notes[torch.argmax(computation_notes_scores)]

                if not in_training:
                    not_removed_notes[note_index] = False
                new_edge_index_dict = self._remove_note(edge_index_dict, note_index, onset_div, duration_div, not_removed_notes)
                new_x_dict = dict(x_dict)
                
                update = Change_('remove', note_index=note_index)

                return new_x_dict, new_edge_index_dict, ts_beats, onset_div, duration_div, update

            case _:
                assert False, 'The operation provided is not implemented. Please provide an operation that is either remove, add, onset, duration or pitch.'

    def _add_note(self, x_dict, edge_index_dict, onset, duration, onset_div, duration_div, feature_vector, not_removed_notes):
        # the new note will be attributed the last index not to mess up the already existing indices
        device = x_dict['note'].device
        new_note_index = len(x_dict['note'])
        new_x_dict = dict(x_dict)
        new_x_dict['note'] = torch.cat((x_dict['note'],feature_vector),0)
        new_x_dict['note'].to(x_dict['note'].device)
        
        # noisy_onset_div = np.concatenate((onset_div, np.array([onset])),axis=None)
        noisy_onset_div = torch.cat((onset_div, torch.tensor([onset],device=device, requires_grad=False)))
        # noisy_duration_div = np.concatenate((duration_div, np.array([duration])),axis=None)
        noisy_duration_div = torch.cat((duration_div, torch.tensor([duration],device=device, requires_grad=False)))

        new_edge_index_dict = self._recompute_edge_dict(
            edge_index_dict, new_note_index, onset, duration, noisy_onset_div, noisy_duration_div, not_removed_notes, remove_previous=False)

        return new_x_dict, new_edge_index_dict, noisy_onset_div, noisy_duration_div
    
    def _remove_note(self, edge_index_dict, note_index, onset_div, duration_div, not_removed_notes, recompute_rest=True): 
        # in order not to mess up the nodes indices in the edge tensors, we keep the note in x_dict, but we isolate
        # it in the graph by removing any edge that is connecting it to other nodes.
        device = edge_index_dict['note','onset','note'].device
        new_edge_index_dict = {}

        for edge_tuple in self.metadata[1]:
            node_type_1, _, node_type_2 = edge_tuple

            match node_type_1, node_type_2:
                case 'note','note':
                    edge_tensor = edge_index_dict[edge_tuple]
                    tensor = torch.ones(2, edge_tensor.size(1), dtype=int, device=device) * note_index
                    edge_contain_note_mask = edge_tensor != tensor  
                    edges_mask = torch.logical_and(edge_contain_note_mask[0,:],edge_contain_note_mask[1,:])
                    first_end_mask = edge_tensor[0,:][edges_mask]
                    second_end_mask = edge_tensor[1,:][edges_mask]
            
                    new_edge_index_dict[edge_tuple] = torch.stack((first_end_mask,second_end_mask))
                
                case 'note',_:
                    edge_tensor = edge_index_dict[edge_tuple]
                    tensor = torch.ones(2, edge_tensor.size(1), dtype=int, device=device) * note_index
                    edge_contain_note_mask = edge_tensor != tensor  
                    edges_mask = edge_contain_note_mask[0,:]
                    first_end_mask = edge_tensor[0,:][edges_mask]
                    second_end_mask = edge_tensor[1,:][edges_mask]
            
                    new_edge_index_dict[edge_tuple] = torch.stack((first_end_mask,second_end_mask))

                case _,'note':
                    edge_tensor = edge_index_dict[edge_tuple]
                    tensor = torch.ones(2, edge_tensor.size(1), dtype=int, device=device) * note_index
                    edge_contain_note_mask = edge_tensor != tensor  
                    edges_mask = edge_contain_note_mask[1,:]
                    first_end_mask = edge_tensor[0,:][edges_mask]
                    second_end_mask = edge_tensor[1,:][edges_mask]
            
                    new_edge_index_dict[edge_tuple] = torch.stack((first_end_mask,second_end_mask))

                case _,_:
                    continue
        
        if recompute_rest:
            # the rest edges have to be re computed because the removal of a note may break one and calls for new rest edges.
            new_edge_index_dict['note', 'rest','note'] = torch.tensor([[],[]],dtype=int,device=device)
            # new_edge_index_dict['note', 'rest_rev','note'] = torch.tensor([[],[]],dtype=int,device=device)
            onset_div_c = onset_div.clone().detach().cpu()
            duration_div_c = duration_div.clone().detach().cpu()
            end_times = onset_div_c + duration_div_c
            for et in np.sort(np.unique(end_times))[:-1]:
            # for et in torch.unique(end_times, sorted=True):
                if et not in onset_div_c:
                    scr = np.where(end_times == et)[0]
                    diffs = onset_div_c - et
                    tmp = np.where(diffs > 0, diffs, np.inf)
                    dst = np.where(tmp == tmp.min())[0]
                    for i in scr:
                        for j in dst:
                            if (not_removed_notes[i]) & (not_removed_notes[j]):
                                new_edge = torch.tensor([[i],[j]], dtype=int,device=device)
                                if i == j:
                                    new_edge_index_dict['note','rest','note'] = torch.cat((new_edge_index_dict['note','rest','note'], new_edge),1)
                                else:
                                    new_edge_rev = torch.tensor([[j],[i]], dtype=int, device=device)
                                    new_edge_index_dict['note','rest','note'] = torch.cat((new_edge_index_dict['note','rest','note'], new_edge, new_edge_rev),1)
                                # new_edge_index_dict['note','rest_rev','note'] = torch.cat((new_edge_index_dict['note','rest_rev','note'], new_edge_rev),1)

        return new_edge_index_dict
    
    def _update_pitch(self, new_pitch, note_index, x_dict):
        new_x_dict = dict(x_dict)
        for pitch in range(12):
            new_x_dict['note'][note_index][1+pitch] = float(pitch==new_pitch)
        return new_x_dict
    
    def _update_onset(self, onset, note_index, onset_div, duration_div, edge_index_dict, not_removed_notes):
        noisy_onset_div = copy.deepcopy(onset_div)
        noisy_onset_div[note_index] = onset

        new_edge_index_dict = self._recompute_edge_dict(
            edge_index_dict, note_index, int(onset), int(duration_div[note_index]), noisy_onset_div, duration_div, not_removed_notes)

        return new_edge_index_dict, noisy_onset_div
    
    def _update_duration(self, duration, note_index, ts_beats, onset_div, duration_div, x_dict, edge_index_dict, not_removed_notes):
        noisy_duration_div = copy.deepcopy(duration_div)
        noisy_duration_div[note_index] = duration

        new_x_dict = dict(x_dict)
        new_x_dict['note'][note_index][0] = 1-tanh(duration/ts_beats[note_index])

        new_edge_index_dict = self._recompute_edge_dict(
            edge_index_dict, note_index, int(onset_div[note_index]), int(duration), onset_div, noisy_duration_div, not_removed_notes)

        return new_x_dict, new_edge_index_dict, noisy_duration_div

    def _recompute_edge_dict(self, edge_index_dict, note_index, onset, duration, onset_div, duration_div, not_removed_notes, remove_previous=True):
        # we remove the edges using the _remove_note method
        device = edge_index_dict['note','onset','note'].device

        if remove_previous:
            new_edge_index_dict = self._remove_note(edge_index_dict, note_index, onset_div, duration_div, not_removed_notes, recompute_rest=False)
        else:
            new_edge_index_dict = dict(edge_index_dict)
        
        onset_div_c = onset_div.clone().detach().cpu()
        duration_div_c = duration_div.clone().detach().cpu()
        for j in np.where((onset_div_c == onset))[0]:
            if not_removed_notes[j] and not_removed_notes[note_index]:
                new_edge = torch.tensor([[note_index],[j]], dtype=int,device=device)
                if j == note_index:
                    new_edge_index_dict['note','onset','note'] = torch.cat((new_edge_index_dict['note','onset','note'], new_edge),1)
                else:
                    new_edge_rev = torch.tensor([[j],[note_index]], dtype=int,device=device)
                    new_edge_index_dict['note','onset','note'] = torch.cat((new_edge_index_dict['note','onset','note'], new_edge, new_edge_rev),1)

        for j in np.where((onset_div_c == onset + duration))[0]:
            if not_removed_notes[j] and not_removed_notes[note_index]:
                new_edge = torch.tensor([[note_index],[j]], dtype=int,device=device)
                if j == note_index:
                    new_edge_index_dict['note','consecutive','note'] = torch.cat((new_edge_index_dict['note','consecutive','note'], new_edge),1)
                else:
                    new_edge_rev = torch.tensor([[j],[note_index]], dtype=int,device=device)
                    new_edge_index_dict['note','consecutive','note'] = torch.cat((new_edge_index_dict['note','consecutive','note'], new_edge, new_edge_rev),1)
                
                # new_edge_index_dict['note','consecutive_rev','note'] = torch.cat((new_edge_index_dict['note','consecutive_rev','note'], new_edge_rev),1)

        for j in np.where((onset < onset_div_c) & (onset + duration > onset_div_c))[0]:
            if not_removed_notes[j] and not_removed_notes[note_index]:
                new_edge = torch.tensor([[note_index],[j]], dtype=int,device=device)
                if j == note_index:
                    new_edge_index_dict['note','during','note'] = torch.cat((new_edge_index_dict['note','during','note'], new_edge),1)
                else:
                    new_edge_rev = torch.tensor([[j],[note_index]], dtype=int,device=device)
                    new_edge_index_dict['note','during','note'] = torch.cat((new_edge_index_dict['note','during','note'], new_edge, new_edge_rev),1)
                # new_edge_index_dict['note','during_rev','note'] = torch.cat((new_edge_index_dict['note','during_rev','note'], new_edge_rev),1)
        
        # the rest edges have to be re computed because the new note may break some old rest edges
        # there is much likely a clever way of updating by looking for the closest onsets and only reseting rest edges for those,
        # it is one idea for optimizing.
        new_edge_index_dict['note', 'rest','note'] = torch.tensor([[],[]],dtype=int, device=device)
        # new_edge_index_dict['note', 'rest_rev','note'] = torch.tensor([[],[]],dtype=int, device=device)
        end_times = onset_div_c + duration_div_c
        for et in np.sort(np.unique(end_times))[:-1]:
        # for et in torch.unique(end_times, sorted=True):
            if et not in onset_div_c:
                scr = np.where(end_times == et)[0]
                diffs = onset_div_c - et
                tmp = np.where(diffs > 0, diffs, np.inf)
                dst = np.where(tmp == tmp.min())[0]
                for i in scr:
                    for j in dst:
                        if (not_removed_notes[i]) & (not_removed_notes[j]):
                            new_edge = torch.tensor([[i],[j]], dtype=int, device=device)
                            if i == j:
                                new_edge_index_dict['note','rest','note'] = torch.cat((new_edge_index_dict['note','rest','note'], new_edge),1)
                            else:
                                new_edge_rev = torch.tensor([[j],[i]], dtype=int, device=device)
                                new_edge_index_dict['note','rest','note'] = torch.cat((new_edge_index_dict['note','rest','note'], new_edge, new_edge_rev),1)
                            # new_edge_index_dict['note','rest_rev','note'] = torch.cat((new_edge_index_dict['note','rest_rev','note'], new_edge_rev),1)

        return new_edge_index_dict

class EncodingGNN_(torch.nn.Module):
    # the encoding module class adapted from graphmuse encoding models. 

    def __init__(self, metadata, input_channels, hidden_channels, num_layers, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropout)
        self.edge_types = metadata[1]
        self.convs.append(
            HeteroConv(
                {
                    edge_type: SAGEConv(input_channels, hidden_channels, normalize=True, project=True)
                    for edge_type in self.edge_types
                }, aggr='mean')
        )
        for _ in range(num_layers-1):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv(hidden_channels, hidden_channels)
                    for edge_type in self.edge_types
                }, aggr='mean')
            self.convs.append(conv)
            self.layer_norms.append(torch.nn.LayerNorm(hidden_channels))

    def forward(self, x_dict, edge_index_dict, neighbor_mask_node=None, neighbor_mask_edge=None):

        for i, conv in enumerate(self.convs[:-1]):
            if not neighbor_mask_edge is None and not neighbor_mask_node is None:
                x_dict, edge_index_dict, _ = trim_to_layer(
                    layer=self.num_layers - i,
                    neighbor_mask_node=neighbor_mask_node,
                    neighbor_mask_edge=neighbor_mask_edge,
                    x=x_dict,
                    edge_index=edge_index_dict,
                )

            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: self.layer_norms[i](x) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        if not neighbor_mask_edge is None and not neighbor_mask_node is None:
            x_dict, edge_index_dict, _ = trim_to_layer(
                layer=1,
                neighbor_mask_node=neighbor_mask_node,
                neighbor_mask_edge=neighbor_mask_edge,
                x=x_dict,
                edge_index=edge_index_dict,
            )
        x_dict = self.convs[-1](x_dict, edge_index_dict)        
        return x_dict
