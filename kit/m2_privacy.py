"""Reject private identifiers in the exact public export scope."""
import hashlib
import re
from pathlib import Path

# Digests of normalized complete tokens; no identifying names or storage tokens ship.
PRIVATE_TOKEN_SHA256 = frozenset(('0bca6f21d009136a49d2b893516d0ddcd85fa08ac64405ef5808e95e70295496', '0c32d5421d5b4bbb52c35493f38652dce7b9c9f80aa79a5d88937832a2cd1b4b', '4772e89b183820d42e2a233465fd2a44aa81551d32e80024e8615bcae47fd0e2', '50bf4c9269f1f982e9310ad430723e1f90018cffab77dab4b960af5e0bb692cf', 'c994ced70b6ec802a8c03e58088a52fd3ffa9f446d1728b2d928f4f4487b9ef3'))


# Public source-document emails are scientific data, exempt only at exact frozen bytes.
PUBLIC_EMAIL_FILES = {'kit/resources/c/contract_nli/panel.jsonl': 'f7e074914a8ea1257e6dfc91a8c1a1c4098b41d785203ea3ccceca705975e170', 'kit/resources/c/contract_nli/test.jsonl': '582413faf8dbe5c0048d7bd82567a75006c8befdcdaa06c8e70dd6fd957e7254', 'kit/resources/c/contract_nli/train.jsonl': '21373e080a593c6865ece5801ef92efb17a590c6cc4ea8fbf8c90df6a025d641'}

def patterns():
    return [re.compile(r'(?<![\w.])/(?:users|home)/(?!<)',re.I),
            re.compile(r'/private/(?:tmp)(?:/|\b)',re.I),
            re.compile(r'/opt/(?:anaconda)[^/\s]*',re.I),
            re.compile(r'[A-Za-z0-9.!#$%&*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:[.][A-Za-z0-9-]+)*[.][A-Za-z]{2,}'),
            re.compile(r'\bh100-(?!(?:hours?|days?|class|sxm\d*(?:-\d+gb)?|nvl|\d+gb)\b)[A-Za-z0-9][A-Za-z0-9_.-]*',re.I),
            re.compile(r'\bip-\d+-\d+-\d+-\d+(?:[.][A-Za-z0-9.-]+)?'),
            re.compile(r'"(?:job_id|jobid|slurm_job_id)"\s*:\s*"?\d+',re.I),
            re.compile(r'\bjob[\s#:=]*(?:id[\s#:=]*)?\d{5,}',re.I)]


def public_paths(root):
    """Select the publisher's source files, or every file in an existing export."""
    root=Path(root);publisher=root/'scripts/publish_kit.py'
    if not publisher.exists():return (p for p in root.rglob('*') if p.is_file())
    import ast
    names=[];runtime=[]
    tree=ast.parse(publisher.read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TESTS' for t in node.targets):names=list(ast.literal_eval(node.value))
        if isinstance(node,ast.AugAssign) and isinstance(node.target,ast.Name) and node.target.id=='TESTS':names.extend(ast.literal_eval(node.value))
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RUNTIME_FILES' for t in node.targets):runtime=list(ast.literal_eval(node.value))
    return [p for p in (root/'kit').rglob('*') if p.is_file()]+[root/'tests'/name for name in names]+[root/name for name in runtime]


def scan(root,paths=None):
    violations=[]
    rules=patterns();root=Path(root)
    for path in public_paths(root) if paths is None else paths:
        if not path.is_file() or path.is_symlink() or '__pycache__' in path.parts:continue
        try:text=path.read_text()
        except UnicodeError:continue
        matches=[]
        public_email_file=PUBLIC_EMAIL_FILES.get(str(path.relative_to(root)))==hashlib.sha256(path.read_bytes()).hexdigest()
        for index,rule in enumerate(rules):
            for match in rule.finditer(text):
                if '\n' in match.group():continue
                if index==3 and (public_email_file or match.group().lower().endswith('@example.com')):continue
                matches.append(match)
        for token in re.finditer(r'[\w]+(?:[-_][\w]+)*',text):
            candidates={token.group(),*re.split(r'[-_]',token.group())}
            for candidate in candidates:
                normal=re.sub(r'[^a-z0-9]','',candidate.lower())
                if hashlib.sha256(normal.encode()).hexdigest() in PRIVATE_TOKEN_SHA256:
                    matches.append(token);break
        for number in sorted({text.count('\n',0,match.start())+1 for match in matches}):
            violations.append(f'{path.relative_to(root)}:{number}')
    return violations


if __name__=='__main__':
    failures=scan(Path(__file__).resolve().parents[1])
    if failures:raise SystemExit('private identifiers in public files: '+', '.join(failures))
    print('public privacy scan PASS (entire publisher scope)')
