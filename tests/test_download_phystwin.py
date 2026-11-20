import importlib.util
import json
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest
import zipfile


spec = importlib.util.spec_from_file_location('download_phystwin', Path(__file__).resolve().parents[1] / 'tools/download_phystwin.py')
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class ExtractionTests(unittest.TestCase):
    def test_extract_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'data.zip'
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('data/example.txt', 'measurement')
            result = downloader.extract(archive, root, 'data')
            self.assertEqual(result, {'files': 1, 'expanded_bytes': 11})
            self.assertEqual((root / 'data/example.txt').read_text(), 'measurement')
            with self.assertRaises(FileExistsError):
                downloader.extract(archive, root, 'data')

    def test_reject_traversal_and_wrong_root(self):
        for member in ['data/../../escape', '/data/absolute', 'other/file', 'data/..\\escape']:
            with self.subTest(member=member), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = root / 'data.zip'
                with zipfile.ZipFile(archive, 'w') as stream:
                    stream.writestr(member, 'bad')
                with self.assertRaises(ValueError):
                    downloader.extract(archive, root, 'data')
                self.assertFalse((root / 'data').exists())


class DownloadTests(unittest.TestCase):
    def test_verified_local_archive_and_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / '.downloads'
            cache.mkdir()
            archive = cache / 'data.zip'
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('data/example.txt', 'measurement')
            release = root / 'release.json'
            release.write_text(json.dumps(dict(repository='example/PhysTwin', revision='pinned', license='mit',
                archives={'data.zip': dict(bytes=archive.stat().st_size, sha256=downloader.sha256(archive))})))
            with patch.object(downloader, 'RELEASE', release):
                downloader.download(root, ['data.zip'])
                record = cache / 'data.zip.json'
                before = record.read_bytes()
                downloader.download(root, ['data.zip'])
                self.assertEqual(record.read_bytes(), before)
                self.assertEqual(json.loads(before)['revision'], 'pinned')
                self.assertEqual((root / 'data/example.txt').read_text(), 'measurement')

    def test_reject_corrupted_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / '.downloads'
            cache.mkdir()
            (cache / 'data.zip').write_bytes(b'corrupt')
            release = root / 'release.json'
            release.write_text(json.dumps(dict(repository='example/PhysTwin', revision='pinned',
                archives={'data.zip': dict(bytes=7, sha256='0'*64)})))
            with patch.object(downloader, 'RELEASE', release):
                with self.assertRaisesRegex(ValueError, 'checksum/size mismatch'):
                    downloader.download(root, ['data.zip'])
            self.assertFalse((root / 'data').exists())


if __name__ == '__main__':
    unittest.main()
