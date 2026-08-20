"""
Unit tests for Backup API endpoints (SQLite/rsync only).
"""

import json
import unittest
from unittest.mock import Mock, patch

from ethoscope_node.api.backup_api import BackupAPI


class TestBackupAPI(unittest.TestCase):
    """Test suite for BackupAPI class (SQLite only)."""

    def setUp(self):
        self.mock_server = Mock()
        self.mock_server.app = Mock()
        self.mock_server.device_scanner = Mock()
        self.mock_server.logger = Mock()
        self.api = BackupAPI(self.mock_server)

    def test_init(self):
        self.assertIsNotNone(self.api._backup_cache)
        self.assertIsNone(self.api._backup_cache["data"])

    def test_register_routes(self):
        route_calls = []

        def mock_route(path, method):
            def decorator(func):
                route_calls.append((path, method, func.__name__))
                return func
            return decorator

        self.api.app.route = mock_route
        self.api.register_routes()
        self.assertEqual(len(route_calls), 1)
        self.assertEqual(route_calls[0][0], "/backup/status")

    @patch("ethoscope_node.api.backup_api.time.time")
    @patch.object(BackupAPI, "_fetch_backup_service_status")
    @patch.object(BackupAPI, "_get_devices_backup_summary")
    def test_get_backup_status_success(self, mock_get_devices, mock_fetch_status, mock_time):
        mock_time.return_value = 1000.0
        rsync_status = {
            "current_device": "ETHOSCOPE_002",
            "current_file": "video_002.h264",
        }
        mock_fetch_status.return_value = rsync_status
        mock_get_devices.return_value = {}

        result = self.api._get_backup_status()
        result_dict = json.loads(result)

        self.assertIn("services", result_dict)
        self.assertIn("rsync_backup", result_dict["services"])
        self.assertTrue(result_dict["services"]["rsync_backup"]["available"])
        self.assertEqual(mock_fetch_status.call_count, 1)

    def test_fetch_backup_service_status_success(self):
        with patch("ethoscope_node.api.backup_api.urllib.request.urlopen") as mock_urlopen:
            mock_response = Mock()
            mock_response.read.return_value = json.dumps({"current_device": "dev"}).encode()
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = self.api._fetch_backup_service_status(8093, "Rsync")
            self.assertIn("current_device", result)

    def test_fetch_backup_service_status_error(self):
        with patch("ethoscope_node.api.backup_api.urllib.request.urlopen", side_effect=Exception("fail")):
            result = self.api._fetch_backup_service_status(8093, "Rsync")
            self.assertIn("error", result)

    def test_get_processing_devices_rsync(self):
        rsync_status = {
            "devices": {
                "dev_001": {"name": "ETHOSCOPE_001", "processing": True},
            }
        }
        result = self.api._get_processing_devices(rsync_status)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["service"], "rsync")

    def test_get_devices_backup_summary_offline(self):
        self.api.device_scanner.get_all_devices_info.return_value = {
            "dev_001": {"status": "offline", "databases": {}}
        }
        result = self.api._get_devices_backup_summary([])
        self.assertIn("dev_001", result)
        self.assertEqual(result["dev_001"]["overall_status"], "no_backups")
