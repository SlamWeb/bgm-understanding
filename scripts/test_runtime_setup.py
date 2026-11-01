"""Installer safety invariants without downloads, environments, or model imports."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from setup_runtime import safe_target, download
from runtime_paths import default_runtime_root


class RuntimeSetupTests(unittest.TestCase):
    def test_manifest_cannot_write_outside_requested_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                safe_target(Path(folder), '../outside/model.pt')

    def test_existing_wrong_asset_is_preserved_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); target=root/'model.pt'; target.write_bytes(b'original')
            asset={'path':'model.pt','bytes':1,'sha256':'0'*64,'url':'https://invalid.example/model.pt'}
            with patch('setup_runtime.urllib.request.urlopen') as network:
                with self.assertRaises(ValueError):download(root,asset)
                network.assert_not_called()
            self.assertEqual(target.read_bytes(),b'original')

    def test_verified_cached_asset_is_reused_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'model.pt').write_bytes(b'verified')
            asset={'path':'model.pt','bytes':8,'sha256':hashlib.sha256(b'verified').hexdigest()}
            with patch('setup_runtime.urllib.request.urlopen') as network:
                download(root,asset)
                network.assert_not_called()

    def test_explicit_runtime_environment_overrides_machine_default(self):
        with patch.dict('os.environ',{'BGM_ANALYSIS_ROOT':'custom-runtime'}):
            self.assertEqual(default_runtime_root(),Path('custom-runtime'))


if __name__=='__main__':unittest.main()
