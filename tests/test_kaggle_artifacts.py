"""Dependency-free validation of the portable artifact and embedded evaluator."""
import ast
import base64
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class PortableNotebookTests(unittest.TestCase):
    def test_notebook_cells_and_embedded_sources(self):
        nb = json.loads((ROOT/'kaggle'/'deepweeds_kaggle.ipynb').read_text(encoding='utf-8'))
        self.assertEqual(nb['nbformat'], 4)
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if source.startswith('%%writefile '): source = source.split('\n', 1)[1]
                ast.parse(source)
        bootstrap = ''.join(nb['cells'][3]['source'])
        tree = ast.parse(bootstrap)
        blob = next(n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == 'b64decode')
        sources = json.loads(base64.b64decode(blob))
        visible_train = next(''.join(cell['source']).split('\n',1)[1] for cell in nb['cells']
                             if ''.join(cell['source']).startswith('%%writefile '))
        self.assertEqual(visible_train, sources['lab_code/train.py'])
        self.assertEqual(sources['eval.py'], (ROOT/'eval.py').read_text(encoding='utf-8'))
        for file in (ROOT/'kaggle'/'code').glob('*.py'):
            self.assertEqual(sources['lab_code/'+file.name], file.read_text(encoding='utf-8'))
            ast.parse(sources['lab_code/'+file.name])
            self.assertNotIn('NotImplementedError', sources['lab_code/'+file.name])

if __name__ == '__main__': unittest.main()
