import partitura as pt
from utils.graph import create_graph_for_score
from utils.score import visualize_explanation_files
from musical_noise_explainer import MNExplainer
from models.cadence_model import CadencePLModel, PitchEncoder
import torch
import os

"""
In this file, we will test our explainer by importing a score, creating its graph, loading a node level classification model
(cadence detection model) and producing counterfactual explanations of an accurate prediction from the model on the graph.
"""

"""First section : Importing what is necessary"""

file_path = os.path.dirname(os.path.abspath(__file__))
score_path = os.path.join(file_path, 'cadence_xml_datasets/data/mozart_piano_sonatas/K280-2.musicxml')
score_name = score_path.split("/")[-1]
pitch_encoder = PitchEncoder()
score = [pt.load_musicxml(score_path), score_name]

#We will store in a 'explain_files' folder all the mei and json files necessary to visualize our graphs and their corresponding music score.

os.makedirs(os.path.join(file_path, 'explain_files'), exist_ok=True)

pt.save_mei(score[0], os.path.join(file_path, 'explain_files', score[1] + '.mei'))


graph, feat_names = create_graph_for_score(score[0], include_cadence=True, include_divs_pq=True, include_id=True, include_ts_beats=True, pitch_encoder=pitch_encoder)
torch.save(graph, os.path.join(file_path, 'explain_files', score[1] + '_graph'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
instance_data = graph
metadata = graph.metadata()
#model = torch.load('saved_models/hgnntestmodel', weights_only=False)
model = CadencePLModel.load_from_checkpoint(os.path.join(file_path, "models/checkpoints/weights.ckpt")).module
model.eval()

model.to(device)
instance_data.to(device)

"""Finding an accurate prediction of the loaded model in the instance_data graph"""

def finding_accurate_predictions(data, model, target, count_label_zero=False, **kwargs):
    x_dict = data.x_dict
    x_dict["pitch_spelling"] = data["note"].pitch_spelling
    accurate = model(x_dict, data.edge_index_dict, **kwargs).argmax(dim=1) == target
    if not count_label_zero:
        accurate = torch.logical_and(accurate, target > 0)
    return accurate

accurate_positive_cadence_mask = finding_accurate_predictions(instance_data, model, instance_data['note'].cadences)
first_cad = 0
for j in range(len(accurate_positive_cadence_mask)):
    if accurate_positive_cadence_mask[j] > 0:
        first_cad = j
        break

"""Explaining"""

num_feat = instance_data['note'].x.size(dim=1)
explained_predictions_indices = [i for i in range(len(accurate_positive_cadence_mask)) if accurate_positive_cadence_mask[i] == 1]
explainer = MNExplainer(model, metadata, num_feat, 'node', num_layers=model.num_layers, epochs=150, balance_factor=2.0)
desired_classification = 0
target=explained_predictions_indices[0]
target=274

explanation, _ = visualize_explanation_files(
    score[0],
    target,
    explainer,
    ['onset', 'duration', 'add'],
    desired_classification,
    score_name=score[1],
    device=device
)

for i, g in enumerate(explanation):
    pred = model(g.x_dict, g.edge_index_dict).argmax(dim=1)[target]
    print(f'For explanation number {i} : target prediction classification is {pred}')
