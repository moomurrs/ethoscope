"""
Backup API Module

Handles backup system management by aggregating status from rsync backup service.
"""

import json
import time
import urllib.request

from .base import BaseAPI, error_decorator


class BackupAPI(BaseAPI):
    """API endpoints for backup system management."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Simple cache for backup status with longer TTL for production
        self._backup_cache = {
            "data": None,
            "timestamp": 0,
            "ttl": 300,  # Cache for 5 minutes (was 60 seconds)
        }

    def register_routes(self):
        """Register backup-related routes."""
        self.app.route("/backup/status", method="GET")(self._get_backup_status)

    @error_decorator
    def _get_backup_status(self):
        """Get backup status aggregated from backup services (SQLite/rsync only)."""
        self.set_json_response()

        # Check cache first
        current_time = time.time()
        if (
            self._backup_cache["data"] is not None
            and current_time - self._backup_cache["timestamp"]
            < self._backup_cache["ttl"]
        ):
            return self._backup_cache["data"]

        # Fetch status from rsync backup service only (SQLite)
        rsync_status = self._fetch_backup_service_status(8093, "Rsync")

        # Determine which devices are currently being processed
        processing_devices = self._get_processing_devices(rsync_status)

        # Get device-level backup information for home page icons
        devices_backup_info = self._get_devices_backup_summary(processing_devices)

        # Determine service availability
        rsync_available = "error" not in rsync_status

        # Create aggregated response
        aggregated_status = {
            "services": {
                "rsync_backup": {
                    "available": rsync_available,
                    "current_device": self._extract_current_device(rsync_status),
                    "current_file": self._extract_current_file(rsync_status),
                },
            },
            "summary": {
                "rsync_backup_available": rsync_available,
                "services": {
                    "rsync_service_available": rsync_available,
                },
            },
            "devices": devices_backup_info,
            "processing_devices": processing_devices,
            "timestamp": current_time,
        }

        # Cache the result
        result = json.dumps(aggregated_status, indent=2)
        self._backup_cache = {"data": result, "timestamp": current_time, "ttl": 60}

        return result

    def _fetch_backup_service_status(self, port: int, service_name: str):
        """Fetch basic status from a backup daemon."""
        try:
            backup_url = f"http://localhost:{port}/status"
            with urllib.request.urlopen(backup_url, timeout=5) as response:
                data = response.read().decode("utf-8")
                return json.loads(data)
        except Exception as e:
            self.logger.debug(
                f"Failed to get {service_name} backup status from port {port}: {e}"
            )
            return {"error": f"{service_name} backup service unavailable"}

    def _extract_current_device(self, service_status):
        """Extract currently processing device from service status."""
        if "error" in service_status:
            return None

        # Check if this is the simplified format (from reverted backup tools)
        if "current_device" in service_status:
            return service_status["current_device"]

        # Check for processing devices in the array format
        if "processing_devices" in service_status:
            processing = service_status["processing_devices"]
            if processing and len(processing) > 0:
                return processing[0].get("device_name")

        # Legacy format - look for processing devices in full status
        if isinstance(service_status, dict):
            for device_id, device_data in service_status.items():
                if isinstance(device_data, dict) and device_data.get(
                    "processing", False
                ):
                    return device_data.get("name", device_id)

        return None

    def _extract_current_file(self, service_status):
        """Extract currently processing file from service status."""
        if "error" in service_status:
            return None

        # Check if this is the simplified format (from reverted backup tools)
        if "current_file" in service_status:
            return service_status["current_file"]

        # Check for processing devices in the array format
        if "processing_devices" in service_status:
            processing = service_status["processing_devices"]
            if processing and len(processing) > 0:
                return processing[0].get("current_file")

        # Legacy format - look for current file in progress data
        if isinstance(service_status, dict):
            for _device_id, device_data in service_status.items():
                if isinstance(device_data, dict) and device_data.get(
                    "processing", False
                ):
                    progress = device_data.get("progress", {})
                    return progress.get("current_file", progress.get("backup_filename"))

        return None

    def _get_processing_devices(self, rsync_status=None):
        """Get list of all currently processing devices from backup services (SQLite/rsync only)."""
        processing = []

        # Extract all processing devices from rsync service
        if isinstance(rsync_status, dict) and "error" not in rsync_status:
            rsync_devices = rsync_status.get("devices", {})
            for _dev_id, dev_data in rsync_devices.items():
                if isinstance(dev_data, dict) and dev_data.get("processing", False):
                    processing.append(
                        {"service": "rsync", "device": dev_data.get("name", _dev_id)}
                    )

        return processing

    def _get_devices_backup_summary(self, processing_devices=None):
        """Get backup summary for all devices for home page display."""
        devices_backup = {}

        # Build a lookup of which device names are currently being processed
        processing_by_name = {}
        for proc in processing_devices or []:
            name = proc.get("device")
            service = proc.get("service")
            if name:
                processing_by_name.setdefault(name, set()).add(service)

        try:
            # Get list of all devices from the device scanner
            if not self.device_scanner:
                return devices_backup

            devices = self.device_scanner.get_all_devices_info()

            # For each device, get basic backup information
            for device_id, device_info in devices.items():
                try:
                    device_status = device_info.get("status", "offline")
                    has_databases = (
                        "databases" in device_info and device_info["databases"]
                    )

                    if device_status != "offline" and has_databases:
                        from ethoscope_node.backup.helpers import get_device_backup_info

                        device_databases = device_info.get("databases", {})
                        backup_info = get_device_backup_info(
                            device_id, device_databases
                        )
                        backup_status = backup_info.get("backup_status", {})

                        sqlite_info = backup_status.get("sqlite", {})
                        video_info = backup_status.get("video", {})

                        # Check if this device is currently being processed
                        device_name = device_info.get("name", "")
                        active_services = processing_by_name.get(device_name, set())

                        # Calculate overall status (sqlite + video only)
                        available_count = sum(
                            [
                                sqlite_info.get("available", False),
                                video_info.get("available", False),
                            ]
                        )

                        if available_count == 2:
                            overall_status = "success"
                        elif available_count > 0:
                            overall_status = "partial"
                        else:
                            overall_status = "no_backups"

                        # Create structure for home page with required fields
                        device_backup_data = {
                            "backup_types": {
                                "sqlite": {
                                    "available": sqlite_info.get("available", False),
                                    "status": (
                                        "success"
                                        if sqlite_info.get("available", False)
                                        else "not_available"
                                    ),
                                    "processing": "rsync" in active_services,
                                    "size": sqlite_info.get("total_size_bytes", 0),
                                    "last_backup": sqlite_info.get("last_backup", 0),
                                    "files": sqlite_info.get("database_count", 0),
                                    "directory": sqlite_info.get("directory", ""),
                                },
                                "video": {
                                    "available": video_info.get("available", False),
                                    "status": (
                                        "success"
                                        if video_info.get("available", False)
                                        else "not_available"
                                    ),
                                    "processing": "rsync" in active_services,
                                    "size": video_info.get("total_size_bytes", 0),
                                    "last_backup": video_info.get("last_backup", 0),
                                    "files": video_info.get("file_count", 0),
                                    "directory": video_info.get("directory", ""),
                                    "size_human": video_info.get("size_human", ""),
                                },
                            },
                            "overall_status": overall_status,
                        }
                    else:
                        # For offline devices or devices without database info
                        device_backup_data = {
                            "backup_types": {
                                "sqlite": {
                                    "available": False,
                                    "status": "offline",
                                    "processing": False,
                                    "size": 0,
                                    "last_backup": 0,
                                    "files": 0,
                                    "directory": "",
                                },
                                "video": {
                                    "available": False,
                                    "status": "offline",
                                    "processing": False,
                                    "size": 0,
                                    "last_backup": 0,
                                    "files": 0,
                                    "directory": "",
                                    "size_human": "",
                                },
                            },
                            "overall_status": "no_backups",
                        }

                    devices_backup[device_id] = device_backup_data

                except Exception:
                    devices_backup[device_id] = {
                        "backup_types": {
                            "sqlite": {
                                "available": False,
                                "status": "unknown",
                                "processing": False,
                                "size": 0,
                                "last_backup": 0,
                                "files": 0,
                                "directory": "",
                            },
                            "video": {
                                "available": False,
                                "status": "unknown",
                                "processing": False,
                                "size": 0,
                                "last_backup": 0,
                                "files": 0,
                                "directory": "",
                                "size_human": "",
                            },
                        },
                        "overall_status": "unknown",
                    }

        except Exception:
            pass

        return devices_backup
