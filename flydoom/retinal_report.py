"""Render recorded receptor-to-Mi1 paths without claiming physiological fit."""

from collections import Counter
import json
from pathlib import Path


def render(output, mapping, system, recordings):
    lookup = {str(root): i for i, root in enumerate(system['ids'])}
    groups = Counter(r['column_index'] for r in system['receptors'])
    representatives = {}
    for receptor in system['receptors']:
        representatives.setdefault(receptor['column_index'], receptor)
    columns = {r['column_index']: r for r in system['columns']}
    paths = []
    for index, receptor in sorted(representatives.items()):
        ids = [receptor['root_id'], receptor['l1_root_id'], receptor['mi1_root_id']]
        indices = [lookup[root] for root in ids]
        r, l, m = indices
        paths.append({'column_index': index, 'root_ids': ids, 'optical_direction': receptor['optical_direction'],
                      'receptors_in_column': groups[index], 'representative_receptor': ids[0],
                      'contacts': [receptor['receptor_to_l1_contacts'], columns[index]['l1_to_mi1_contacts']],
                      'graded_weights': [float(system['model'].weights[l, r]), float(system['model'].weights[m, l])],
                      'traces': {condition: values[:, indices].tolist() for condition, values in recordings.items()}})
    payload = {'report': json.loads((output / 'report.json').read_text()),
               'mapping': json.loads((mapping / 'report.json').read_text()), 'paths': paths}
    text = json.dumps(payload, allow_nan=False).replace('<', '\\u003c')
    template = Path(__file__).with_name('retinal_report.html').read_text(encoding='utf-8')
    (output / 'index.html').write_text(template.replace('__REPORT_DATA__', text), encoding='utf-8')
