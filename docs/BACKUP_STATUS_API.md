# Unified Backup Status API

## Overview

The Ethoscope node server provides backup status information via rsync:

- **Rsync Backup Daemon** (port 8093) – handles file-based backups (SQLite databases + videos)

## API Endpoint

**GET** `/backup/status`

Returns status information from the rsync backup service.

## Response Format

```json
{
  "rsync_backup": {
    "devices": {
      "device_id": {
        "name": "ETHOSCOPE_XXX", 
        "status": "stopped|running",
        "progress": {
          "status": "success|error",
          "message": "Backup status message"
        },
        "synced": {
          "results": {
            "local_files": 766,
            "directory": "/ethoscope_data/results",
            "disk_usage_bytes": 287646297933,
            "disk_usage_human": "267.9 GB"
          },
          "videos": {
            "local_files": 12529,
            "directory": "/ethoscope_data/videos", 
            "disk_usage_bytes": 544649438682,
            "disk_usage_human": "507.2 GB"
          }
        },
        "processing": false,
        "count": 818,
        "started": 1752137339,
        "ended": 1752137340,
        "metadata": {}
      }
    },
    "disk_usage_summary": {
      "results": {
        "total_files": 29119,
        "total_size_bytes": 11215763841541,
        "total_size_human": "10.2 TB"
      },
      "videos": {
        "total_files": 476102,
        "total_size_bytes": 20696678673241,
        "total_size_human": "18.8 TB"
      }
    }
  },
  "unified_devices": {
    "device_id": {
      "name": "ETHOSCOPE_XXX",
      "status": "stopped|running", 
      "overall_status": "success|partial|error|unknown",
      "rsync_backup": {
        "available": true|false,
        "status": "stopped|running|not_available",
        "progress": {},
        "synced": {},
        "processing": false,
        "count": 0,
        "started": null,
        "ended": null,
        "metadata": {}
      }
    }
  }
}
```

## Overall Status Logic

The `overall_status` field in `unified_devices` is determined as follows:

- **success**: Rsync backup reports success
- **partial**: Rsync backup partially available  
- **error**: Rsync backup reports an error
- **unknown**: Rsync backup status unclear

## Error Handling

If the backup service is unavailable, the response will include:

```json
{
  "rsync_backup": {
    "error": "Rsync backup service unavailable", 
    "service": "rsync_backup"
  },
  "unified_devices": {}
}
```

## Testing

You can test the service directly:

- Rsync backup: `curl http://localhost:8093/status`
- Unified status: `curl http://localhost/backup/status`

## Frontend Changes

The Ethoscope node frontend has been updated to work with the SQLite/rsync backup format:

### Updated JavaScript Functions:

1. **`get_backup_status()`**: Now extracts `unified_devices` and stores service availability flags
2. **`getBackupStatusClass()`**: Determines backup circle colors based on processing state and overall status:
   - **Orange (breathing)**: `processing` - backup currently running
   - **Green**: `success` - rsync backup working
   - **Golden**: `partial` - partial backup  
   - **Red**: `error` - backup failed
   - **Grey**: `unknown` - status unclear
   - **Black**: Service offline

3. **`getBackupStatusTitle()`**: Provides comprehensive tooltip showing:
   - Overall backup status
   - Rsync backup status and message  
   - Data size information from rsync backups

### New Scope Variables:

- `$scope.backup_status`: Contains `unified_devices` for easy device lookup
- `$scope.rsync_backup_available`: Boolean indicating rsync backup daemon availability
- `$scope.backup_service_available`: Boolean indicating service availability
- `$scope.backup_status_full`: Full API response for debugging

## Example Usage

```javascript
// Fetch unified backup status
fetch('/backup/status')
  .then(response => response.json())
  .then(data => {
    const rsyncAvailable = !data.rsync_backup.error;
    
    // Iterate through unified device view
    Object.entries(data.unified_devices).forEach(([deviceId, device]) => {
      console.log(`${device.name}: ${device.overall_status}`);
      
      if (device.rsync_backup.available) {
        console.log(`  Rsync: ${device.rsync_backup.progress.status}`);
        console.log(`  Data: ${device.rsync_backup.synced.results?.disk_usage_human}`);
      }
    });
  });
```
