"""Read-only SHA256 and file-membership verification of the prepared release."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def ignored(path):
    return any(part in {'.git','__pycache__','.venv','venv'} or part.endswith('.egg-info') for part in path.parts)

def main():
    root=Path(__file__).resolve().parents[1]
    manifest=root/'RELEASE_MANIFEST.json'
    try:
        data=json.loads(manifest.read_text(encoding='utf-8'))
        failures=[]
        expected=set()
        for item in data['files']:
            rel=PurePosixPath(item['path'])
            if rel.is_absolute() or '..' in rel.parts or '\\' in item['path'] or ':' in item['path']:
                raise ValueError('Unsafe manifest path')
            if item['path'] in expected:raise ValueError('Duplicate manifest path')
            expected.add(item['path'])
            path=(root/Path(*rel.parts)).resolve()
            path.relative_to(root)
            if not path.is_file() or path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:
                failures.append({'path':item['path'],'reason':'Missing file or byte/hash mismatch'})
        actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and not ignored(p.relative_to(root))}
        unexpected=sorted(actual-expected-{'RELEASE_MANIFEST.json'})
        missing=sorted(expected-actual)
        if missing:failures.append({'missing':missing})
        if unexpected:failures.append({'unexpected':unexpected})
        report={'passed':not failures,'indexed_files':len(expected),'manifest_sha256':digest(manifest),
                'scope':'Packaged file bytes and membership only; no model loading or experiment computation',
                'failures':failures}
    except (OSError,KeyError,TypeError,ValueError) as exc:
        report={'passed':False,'error':str(exc)}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['passed'] else 1

if __name__=='__main__':sys.exit(main())
