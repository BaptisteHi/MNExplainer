import partitura.score as scr
import partitura as pt
from utils.graph import save_pyg_graph_as_json, create_graph_for_score
import time

def _spelling_from_midi_pitch(pitch):
    match pitch:
        case 0:
            return 'C',0
        case 1:
            return 'C',1
        case 2:
            return 'D',0
        case 3:
            return 'D',1
        case 4:
            return 'E',0
        case 5:
            return 'F',0
        case 6:
            return 'F',1
        case 7:
            return 'G',0
        case 8:
            return 'G',1
        case 9:
            return 'A',0
        case 10:
            return 'A',1
        case 11:
            return 'B',0
        case _:
            assert False, 'Failed to recognize pitch - must be an integer between 0 and 11'

def midi_pitch_from_spelling(gm_spelling):
    """
    """
    pitch = 0
    return pitch

def _next_id_available(part : scr.Part):
    max_id = max([int(note.id.split('n')[-1]) for note in part.notes_tied])
    return max_id + 1
        
def update_score(score, change, score_indices, current_next_available) -> scr.Score:
    """
    Updates a score object accordingly to the noise encapsulated in a change object.

    Parameters
    ----------
    score : scr.Score
        the score of the original graph before noising
    change : list (todo : change object)
        the information of the noise
    score_indices : list
        a matching between the notes indices in the score object and in the graph where the noising is performed.

    Returns
    -------

    the updated score object.
    """
    # We assume here that the score contains only one part since we want to visualize it using SMUG-Explain
    part = score[0]
    # sys.setrecursionlimit(10000)
    # part = copy.deepcopy(original_part)
    # sys.setrecursionlimit(1000)
    notes_tied_indexing = [] # could use and update this : 
    # relation between the 'p0nx' id and the position of the related note in the part.notes_tied array.
    # this would allow us to prevent finding the target note object using a loop, we could directly index the right part.notes_tied array.

    match change.operation:
        case 'pitch':
            note_index = change.note_index
            new_pitch = change.pitch
            score_idx = score_indices[note_index]
            note = None
            for no in part.notes_tied:
                if no.id == score_idx:
                    note = no
                    break
            start = note.start
            next_time_point = start.next
            step, alter = _spelling_from_midi_pitch(new_pitch)
            for note_candidate in part.iter_all(cls=scr.Note, start=start, end=next_time_point):
                if (note.octave == note_candidate.octave) & (note.step == note_candidate.step) & (note.alter == note_candidate.alter):
                    note_candidate.alter = alter
                    note_candidate.step = step
            return current_next_available

        case 'onset':
            note_index = change.note_index
            new_onset = change.onset.cpu()
            score_idx = score_indices[note_index]
            for no in part.notes_tied:
                if no.id == score_idx:
                    note = no
                    break

            # creating the new note that will be put to the new onset
            new_start_time_point = part.get_or_add_point(new_onset)
            duration = note.duration
            new_end_time_point = part.get_or_add_point(new_onset + duration)
            new_onset_note = scr.Note(step=note.step, octave=note.octave, alter=note.alter)
            if current_next_available is None:
                current_next_available = _next_id_available(part)
            new_score_id = 'p0n' + str(current_next_available)
            current_next_available += 1
            new_onset_note.id = new_score_id
            new_onset_note.voice = 10
            new_onset_note.staff = note.staff
            new_onset_note.start = new_start_time_point
            new_onset_note.end = new_end_time_point

            """
            for notes in part.iter_all(
                scr.GenericNote,
                start=0,
                end=36,
                include_subclasses=True,
                ):
                    print(notes)
            """
            # removing the previous note
            part.remove(note)

            # adding the note
            part.add(new_onset_note, start=new_onset, end=new_onset + duration)
            
            # updating the mapping between graph nodes and score indices
            score_indices[note_index] = new_score_id
            return current_next_available
            
        case 'duration':
            note_index = change.note_index
            new_duration = change.duration.cpu()
            score_idx = score_indices[note_index]
            for no in part.notes_tied:
                if no.id == score_idx:
                    note = no
                    break

            # creating the new note that will be put to the new onset
            start_time_point = note.start
            new_end_time_point = part.get_or_add_point(start_time_point.t + new_duration)
            new_duration_note = scr.Note(step=note.step, octave=note.octave, alter=note.alter)
            if current_next_available is None:
                current_next_available = _next_id_available(part)
            new_score_id = 'p0n' + str(current_next_available)
            current_next_available += 1
            new_duration_note.id = new_score_id
            new_duration_note.voice = note.voice
            new_duration_note.staff = note.staff
            new_duration_note.start = start_time_point
            new_duration_note.end = new_end_time_point

            """
            for notes in part.iter_all(
                scr.GenericNote,
                start=0,
                end=36,
                include_subclasses=True,
                ):
                    print(notes)
            """
            # removing the previous note
            part.remove(note)

            # adding the note
            part.add(new_duration_note, start=start_time_point.t, end=start_time_point.t + new_duration)
            
            # updating the mapping between graph nodes and score indices
            score_indices[note_index] = new_score_id
            return current_next_available

        case 'add':
            new_note_pitch = change.pitch
            new_note_octave = change.octave
            new_note_onset = change.onset
            new_note_duration = change.duration

            new_start_time_point = part.get_or_add_point(new_note_onset)
            new_end_time_point = part.get_or_add_point(new_note_onset + new_note_duration)
            step, alter = _spelling_from_midi_pitch(new_note_pitch)
            new_note = scr.Note(step=step, octave=new_note_octave, alter=alter)

            if current_next_available is None:
                current_next_available = _next_id_available(part)
            new_note_id = 'p0n' + str(current_next_available)
            current_next_available += 1

            new_note.id = new_note_id
            score_indices.append(new_note_id)
            new_note.start = new_start_time_point
            new_note.end = new_end_time_point
            new_note.staff = 1 + int(new_note_octave < 5)
            new_note.voice = -1

            part.add(new_note, start=new_note_onset, end=new_note_onset + new_note_duration)

            return current_next_available

        case 'remove': 
            return current_next_available

        case _:
            assert False, 'the change does not correspond to a well known operation'

def visualize_explanation_files(score, graph, pitch_encoder, target, explainer, num_expl, desired_classification, path='./explain_files/', score_name = 'score', device='cpu', ret_time=False):
    """
    Produce the files for visualizing the num_expl explanations produced by the provided explainer on the prediction associated
    with the given target and made by the GNN model associated with the explainer.

    Parameters
    ----------
    score : scr.Score
        the score which graph will be fed into the GNN model and then explained
    target : int, tuple or None
        the target of the prediction. Its type sould be adjusted according to the prediction level of the explained model :
        int if the GNN performs on node level, tuple for edge level predictions, and None for graph level.
    explainer : MNExplainer class object
        the explainer
    num_expl : int or a list of strings of operations accepted by the explainer.
        the number of explanations to produce, or the list of operations to perform if the user wants to specify those.
    path (optional) : string
        the path to where the files are stored.
    score_name (optional) : string
        the name of the score, used to name the produced mei files
    device (optional) : string
        the device of the explained model, so the graph produced can be moved accordingly
    """
    ids = graph['note'].id.tolist()
    t_start = time.time()
    explanation, changes, dist = explainer(graph, desired_classification, target=target, num_expl=num_expl, retrieve_changes=True, retrieve_dist=True)
    t_end = time.time()
    current_next_available = None
    graph.to('cpu')
    for i,graphexp in enumerate(explanation):

        # generating mei file
        change = changes[i]
        d = dist[i]
        print(change)
        print(f"Distance from input graph for this explanation : {d}")
        
        if change == None:
            pass
        else:
            current_next_available = update_score(score, change, ids, current_next_available)
            pt.save_mei(score, path + score_name + '_' +  str(i) + 'changes.mei')

        # generating json graph file
        name = 'explanation_number_' +str(i) +'.json'
        indexation = {}
        indexation['id'] = {i : ids[i] for i in range(len(ids))}
        save_pyg_graph_as_json(graphexp, ids, name, extra_info=indexation, path = path)
        
    
    if ret_time:
        time_elapsed = t_end - t_start
        return explanation, changes, dist, time_elapsed

    return explanation, changes, dist
