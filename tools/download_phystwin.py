"""Download the pinned PhysTwin release, verify SHA256, and extract without overwrites."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
import zipfile


RELEASE = Path(__file__).resolve().parents[1] / 'datasets/phystwin-release-v1.json'
DEFAULT_ARCHIVES = ['data.zip', 'experiments_optimization.zip', 'experiments.zip', 'gaussian_output.zip']


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def extract(archive, root, expected_root):
    target = root / expected_root
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        for member in members:
            path = PurePosixPath(member.filename)
            if (path.is_absolute() or '..' in path.parts or '\\' in member.filename
                    or not path.parts or path.parts[0] != expected_root
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError(f'Unsafe or unexpected ZIP member: {member.filename}')
        expanded = sum(member.file_size for member in members)
        if shutil.disk_usage(root).free < expanded + 10 * 1024**3:
            raise OSError('Extraction requires expanded archive size plus 10 GiB free')
        # ZIP CRC is checked while extracting. Publish only a complete directory.
        with tempfile.TemporaryDirectory(prefix='.extract-', dir=root) as staging:
            source.extractall(staging)
            (Path(staging) / expected_root).rename(target)
    return {'files': sum(not m.is_dir() for m in members), 'expanded_bytes': expanded}


def download(root, names):
    release = json.loads(RELEASE.read_text())
    cache = root / '.downloads'
    cache.mkdir(parents=True, exist_ok=True)
    for name in names:
        spec = release['archives'][name]
        archive = cache / name
        record = cache / (name + '.json')
        if record.exists():
            saved = json.loads(record.read_text())
            if (saved['sha256'] != spec['sha256'] or saved['revision'] != release['revision']
                    or not (root / Path(name).stem).is_dir()):
                raise ValueError(f'Inconsistent completed download: {record}')
            print(f'Already extracted: {name} ({record})', flush=True)
            continue
        if (root / Path(name).stem).exists():
            raise FileExistsError(f'Existing unregistered dataset directory: {root / Path(name).stem}')
        url = f"https://huggingface.co/datasets/{release['repository']}/resolve/{release['revision']}/{name}"
        if not archive.exists():
            partial = cache / (name + '.part')
            remaining = max(0, spec['bytes'] - (partial.stat().st_size if partial.exists() else 0))
            if shutil.disk_usage(root).free < remaining + 10 * 1024**3:
                raise OSError('Download requires archive size plus 10 GiB free')
            print(f'Downloading {url}', flush=True)
            subprocess.run(['curl', '--silent', '--show-error', '--fail', '--location', '--retry', '5', '--retry-delay', '3',
                            '--connect-timeout', '30', '--speed-time', '120', '--speed-limit', '1024',
                            '--continue-at', '-', '--output', str(partial), url], check=True)
            if partial.stat().st_size != spec['bytes'] or sha256(partial) != spec['sha256']:
                raise ValueError(f'Archive checksum/size mismatch; preserved for inspection: {partial}')
            partial.rename(archive)
        elif archive.stat().st_size != spec['bytes'] or sha256(archive) != spec['sha256']:
            raise ValueError(f'Archive checksum/size mismatch: {archive}')
        summary = extract(archive, root, Path(name).stem)
        record.write_text(json.dumps(dict(spec, **summary, url=url, revision=release['revision'],
                                         license=release['license'],
                                         extracted_at=datetime.now(timezone.utc).isoformat()), indent=2) + '\n')
        print(f'Verified and extracted: {name}: {summary}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.home() / 'PhysTwin')
    parser.add_argument('--archives', nargs='+', choices=list(json.loads(RELEASE.read_text())['archives']),
                        default=DEFAULT_ARCHIVES)
    args = parser.parse_args()
    download(args.root.expanduser().resolve(), args.archives)


if __name__ == '__main__':
    main()
