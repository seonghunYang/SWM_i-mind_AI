from annoy import AnnoyIndex
import json
import pickle
import annotator

if __name__ == '__main__':
    ann = AnnoyIndex(2048, 'euclidean')
    ann.load('test.annoy')

    with open('ann_idxs_by_id.json', 'r') as f:
        ann_idxs_by_id = json.load(f)

    with open('fused.pkl', 'rb') as f:
        fused = pickle.load(f)

    print(f'------ Exploratory Data Analysis ------')
    print(f'ann.get_n_items(): {ann.get_n_items()}')
    print(f'len(ann_idxs_by_id.keys()): {len(ann_idxs_by_id.keys())}')
    print(f'type(ann.get_item_vector(0): {type(ann.get_item_vector(0))}')
    print(f'len(ann.get_item_vector(0)): {len(ann.get_item_vector(0))}')
    print(f'len(fused): {len(fused)}\n')

    ids = [0]*ann.get_n_items()

    for id, ann_idxs in ann_idxs_by_id.items():
        if 10 < len(ann_idxs):
            for ann_idx in ann_idxs:
                ids[ann_idx] = id

    rids = [id for id in ids if id != 0]

    annotator.to_tsv(ann, ids)
    annotator.explore_fused_by_time(rids, fused)

    print('Program terminated successfully.')

