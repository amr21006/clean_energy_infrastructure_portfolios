from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import pandas as pd


MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
DOC_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"

MONTH_NUMBER = {
    name: number
    for number, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        start=1,
    )
}


def normalize_header(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_identifier(value: object) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip().upper()


def safe_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: object) -> int | None:
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def excel_column_index(reference: str) -> int:
    letters = "".join(character for character in reference if character.isalpha())
    index = 0
    for character in letters.upper():
        index = index * 26 + ord(character) - 64
    return index - 1


def discover_monthly_files(monthly_dir: Path) -> list[tuple[pd.Period, Path]]:
    pattern = re.compile(r"([a-z]+)_generator(\d{4})\.xlsx$", re.IGNORECASE)
    discovered: list[tuple[pd.Period, Path]] = []
    for path in monthly_dir.glob("*_generator*.xlsx"):
        match = pattern.match(path.name)
        if not match:
            continue
        month = MONTH_NUMBER.get(match.group(1).lower())
        if month is None:
            continue
        discovered.append(
            (pd.Period(f"{match.group(2)}-{month:02d}", freq="M"), path)
        )
    discovered.sort(key=lambda item: item[0])
    return discovered


@dataclass
class XlsxSheet:
    name: str
    rows: list[list[object]]

    def records(self) -> Iterator[dict[str, object]]:
        if not self.rows:
            return
        preview_count = min(6, len(self.rows))
        header_index = max(
            range(preview_count),
            key=lambda index: sum(value is not None for value in self.rows[index]),
        )
        headers = [normalize_header(value) for value in self.rows[header_index]]
        for row in self.rows[header_index + 1 :]:
            record = {
                header: row[index] if index < len(row) else None
                for index, header in enumerate(headers)
                if header
            }
            if record:
                yield record


class FastXlsxReader:
    """Read selected sheets directly from OOXML without modifying source files."""

    def __init__(self, path: Path):
        self.path = path

    def read_sheets(self, prefixes: Sequence[str]) -> dict[str, XlsxSheet]:
        with zipfile.ZipFile(self.path) as archive:
            shared_strings = self._read_shared_strings(archive)
            targets = self._sheet_targets(archive, prefixes)
            return {
                name: XlsxSheet(
                    name=name,
                    rows=self._read_worksheet(archive, target, shared_strings),
                )
                for name, target in targets
            }

    @staticmethod
    def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
        if "xl/sharedStrings.xml" not in archive.namelist():
            return []
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        return [
            "".join(text.text or "" for text in item.iter(MAIN_NS + "t"))
            for item in root.findall(MAIN_NS + "si")
        ]

    @staticmethod
    def _sheet_targets(
        archive: zipfile.ZipFile, prefixes: Sequence[str]
    ) -> list[tuple[str, str]]:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relation_targets = {
            relation.attrib["Id"]: relation.attrib["Target"]
            for relation in relations.findall(PKG_REL_NS + "Relationship")
        }
        sheets = workbook.find(MAIN_NS + "sheets")
        if sheets is None:
            return []
        selected: list[tuple[str, str]] = []
        lowered = tuple(prefix.lower() for prefix in prefixes)
        for sheet in sheets:
            name = sheet.attrib["name"]
            if not name.lower().startswith(lowered):
                continue
            target = relation_targets[sheet.attrib[DOC_REL_NS + "id"]].replace("\\", "/")
            if target.startswith("/"):
                target = target.lstrip("/")
            elif not target.startswith("xl/"):
                target = "xl/" + target
            selected.append((name, target))
        return selected

    @staticmethod
    def _read_worksheet(
        archive: zipfile.ZipFile, target: str, shared_strings: Sequence[str]
    ) -> list[list[object]]:
        rows: list[list[object]] = []
        data = io.BytesIO(archive.read(target))
        for _, element in ET.iterparse(data, events=("end",)):
            if element.tag != MAIN_NS + "row":
                continue
            cells: dict[int, object] = {}
            for cell in element.findall(MAIN_NS + "c"):
                column = excel_column_index(cell.attrib.get("r", "A1"))
                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    inline = cell.find(MAIN_NS + "is")
                    value: object = (
                        "".join(
                            text.text or "" for text in inline.iter(MAIN_NS + "t")
                        )
                        if inline is not None
                        else ""
                    )
                else:
                    raw_element = cell.find(MAIN_NS + "v")
                    raw = raw_element.text if raw_element is not None else None
                    if raw is None:
                        value = None
                    elif cell_type == "s":
                        value = shared_strings[int(raw)]
                    elif cell_type == "b":
                        value = raw == "1"
                    else:
                        try:
                            numeric = float(raw)
                            value = int(numeric) if numeric.is_integer() else numeric
                        except ValueError:
                            value = raw
                cells[column] = value
            if cells:
                rows.append([cells.get(index) for index in range(max(cells) + 1)])
            element.clear()
        return rows


def record_value(record: Mapping[str, object], *names: str) -> object:
    for name in names:
        if name in record:
            return record[name]
    return None


def period_from_year_month(year: object, month: object) -> pd.Period | None:
    year_value = safe_int(year)
    month_value = safe_int(month)
    if year_value is None or month_value is None or not 1 <= month_value <= 12:
        return None
    try:
        return pd.Period(f"{year_value:04d}-{month_value:02d}", freq="M")
    except ValueError:
        return None


def technology_group(
    technology: object, energy_source: object = None, prime_mover: object = None
) -> str:
    text = " ".join(
        str(value or "").lower()
        for value in (technology, energy_source, prime_mover)
    )
    source = str(energy_source or "").strip().upper()
    mover = str(prime_mover or "").strip().upper()
    if (
        any(
            phrase in text
            for phrase in (
                "batter",
                "energy storage",
                "flywheel",
                "compressed air",
                "pumped storage",
            )
        )
        or mover in {"BA", "ES", "CE", "FW", "PS"}
    ):
        return "storage"
    if "solar" in text or source == "SUN" or mover in {"PV", "CP"}:
        return "solar"
    if "wind" in text or source == "WND" or mover in {"WT", "WS"}:
        return "wind"
    if "hydro" in text or source == "WAT" or mover == "HY":
        return "hydro"
    if any(phrase in text for phrase in ("biomass", "wood", "geothermal")):
        return "other_renewable"
    if (
        any(phrase in text for phrase in ("natural gas", "landfill gas", "other gases"))
        or source in {"NG", "LFG", "OG"}
    ):
        return "gas"
    if "coal" in text or source in {"BIT", "SUB", "LIG", "WC", "RC"}:
        return "coal"
    if "nuclear" in text or source == "NUC":
        return "nuclear"
    if any(phrase in text for phrase in ("waste", "biogenic")):
        return "other_renewable"
    return "other"


STATUS_PATTERNS: tuple[tuple[str, str], ...] = (
    ("(P)", "P"),
    ("(L)", "L"),
    ("(T)", "T"),
    ("(U)", "U"),
    ("(V)", "V"),
    ("(TS)", "TS"),
    ("(OP)", "O"),
)


def harmonize_status(value: object) -> str:
    text = normalize_header(value).upper()
    if not text:
        return "OTHER"
    for prefix, state in STATUS_PATTERNS:
        if text.startswith(prefix):
            return state
    if "CONSTRUCTION COMPLETE" in text or "TEST" in text:
        return "TS"
    if "MORE THAN 50" in text:
        return "V"
    if "UNDER CONSTRUCTION" in text:
        return "U"
    if "APPROVALS RECEIVED" in text:
        return "T"
    if "APPROVALS PENDING" in text:
        return "L"
    if "NOT INITIATED" in text:
        return "P"
    if "OPERAT" in text:
        return "O"
    return "OTHER"


def sector_group(value: object) -> str:
    text = normalize_header(value).lower()
    if "electric utility" in text:
        return "electric_utility"
    if "ipp" in text:
        return "independent_power_producer"
    if "industrial" in text:
        return "industrial"
    if "commercial" in text:
        return "commercial"
    return "other_or_unknown"
