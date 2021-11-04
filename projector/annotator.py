import csv
import os
from tqdm import tqdm

def to_tsv(annoy_index, labels):
    print(f'------ Making .tsv file for tensorflow projecter ------')
    
    with open('data.tsv', 'w', encoding='utf-8', newline='') as f:
        tw = csv.writer(f, delimiter='\t')
        
        for i in range(annoy_index.get_n_items()):
            if labels[i] != 0:
                tw.writerow(annoy_index.get_item_vector(i))

    with open('labels.tsv', 'w', encoding='utf-8', newline='') as f:
        tw = csv.writer(f, delimiter='\t')

        for label in labels:
            if label != 0:
                tw.writerow(label)

def explore_fused_by_time(labels, fused):
    print(f'fused: {fused}') # list of (nid, combined_id)
    print(f'Exploring fused by time...')

    os.makedirs('./fused_by_time', exist_ok=True)

    for i in tqdm(range(len(fused)), desc='Exploring'):
        nid, combined_id = fused[i]

        for j, label in enumerate(labels):
            if int(label) == int(nid):
                labels[j] = combined_id

        # switch done.
        with open(f'./fused_by_time/{i}.tsv', 'w', encoding='utf-8', newline='') as f:
            tw = csv.writer(f, delimiter='\t')

            for label in labels:
                tw.writerow([label])