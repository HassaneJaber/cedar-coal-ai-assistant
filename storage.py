"""Small durable storage adapter. No customer details or secrets in logs."""
import csv
import json
import logging
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from filelock import FileLock

ROOT = Path(__file__).resolve().parent
FIELDS = {
    "Bookings": ["timestamp", "name", "phone", "date", "time", "party_size", "seating", "notes"],
    "Handoffs": ["timestamp", "name", "phone", "reason"],
}

def safe_cell(value):
    text = str(value)
    # Prevent formula injection when CSV files are opened in a spreadsheet.
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text

class Storage:
    def __init__(self, data_dir=None, local_only=False):
        self.data_dir = Path(data_dir or ROOT / "data")
        self.sheet = None
        credentials = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
        sheet_id = os.getenv("GOOGLE_SHEET_ID", "")
        if credentials and sheet_id and not local_only:
            import gspread
            try:
                info = json.loads(credentials) if credentials.lstrip().startswith("{") else json.loads(Path(credentials).read_text("utf-8"))
                self.sheet = gspread.service_account_from_dict(info).open_by_key(sheet_id)
                for title, fields in FIELDS.items():
                    try:
                        tab = self.sheet.worksheet(title)
                    except gspread.WorksheetNotFound:
                        tab = self.sheet.add_worksheet(title, 1000, len(fields))
                    first = tab.row_values(1)
                    if not first:
                        tab.append_row(fields, value_input_option="RAW")
                    elif first != fields:
                        raise ValueError("Sheet headers do not match")
            except Exception:
                # Configured-but-broken Sheets must not silently create divergent records.
                raise RuntimeError("Google Sheets setup failed. Check credentials, sharing and headers.") from None
        self.backend = "Google Sheets" if self.sheet else "local CSV"
        logging.getLogger(__name__).info("Storage backend: %s", self.backend)

    def save(self, kind, values, dry_run=False):
        if dry_run:
            return {"ok": True, "dry_run": True, "backend": "dry-run"}
        row = {"timestamp": datetime.now(ZoneInfo("Asia/Beirut")).isoformat(), **values}
        fields = FIELDS[kind]
        if self.sheet:
            self.sheet.worksheet(kind).append_row([str(row.get(k, "")) for k in fields], value_input_option="RAW")
        else:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            path = self.data_dir / (kind.lower() + ".csv")
            with FileLock(str(path) + ".lock"):
                new = not path.exists() or path.stat().st_size == 0
                with path.open("a", newline="", encoding="utf-8-sig") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    if new:
                        writer.writeheader()
                    writer.writerow({k: safe_cell(row.get(k, "")) for k in fields})
        return {"ok": True, "backend": self.backend}
