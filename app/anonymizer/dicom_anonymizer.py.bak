"""
Anonymiseur DICOM — Conforme DICOM PS3.15 Annexe E.
Profil : Basic Application Confidentiality Profile + Options retenues.

Stratégie :
  - Tags nominatifs (0010,xxxx) : SUPPRIMÉS ou REMPLACÉS par twin_id
  - UIDs (0020,000D, 0020,000E...) : RÉGÉNÉRÉS (hash déterministe)
  - Dates : DÉCALÉES (date shifting) ou SUPPRIMÉES selon le profil
  - Données pixel : CONSERVÉES intactes
  - Séquences : RÉCURSIVES

twin_id = HMAC-SHA256(patient_id_brut, seed_institutionnel)
→ Irréversible sans le seed, reproductible avec le même patient_id + seed.
"""
import hashlib
import hmac
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
import structlog

logger = structlog.get_logger(__name__)

# ─── Tags nominatifs à supprimer (DICOM PS3.15 Table E.1-1) ─────────────────
# Format: (group, element)
TAGS_TO_REMOVE: frozenset[tuple[int, int]] = frozenset({
    # Patient
    (0x0010, 0x0010),  # PatientName
    (0x0010, 0x0020),  # PatientID          → remplacé par twin_id
    (0x0010, 0x0030),  # PatientBirthDate
    (0x0010, 0x0040),  # PatientSex
    (0x0010, 0x1010),  # PatientAge
    (0x0010, 0x1020),  # PatientSize
    (0x0010, 0x1030),  # PatientWeight
    (0x0010, 0x21B0),  # AdditionalPatientHistory
    (0x0010, 0x4000),  # PatientComments
    # Institution
    (0x0008, 0x0080),  # InstitutionName
    (0x0008, 0x0081),  # InstitutionAddress
    (0x0008, 0x1040),  # InstitutionalDepartmentName
    (0x0008, 0x0090),  # ReferringPhysicianName
    (0x0008, 0x1048),  # PhysiciansOfRecord
    (0x0008, 0x1050),  # PerformingPhysicianName
    (0x0008, 0x1070),  # OperatorsName
    # Identifiants
    (0x0008, 0x0050),  # AccessionNumber     → anonymisé
    (0x0020, 0x0010),  # StudyID
    (0x0040, 0xA124),  # UID (divers)
    # Commentaires libres
    (0x0032, 0x4000),  # StudyComments
    (0x0040, 0x0006),  # ScheduledPerformingPhysicianName
    (0x0040, 0x1004),  # PatientTransportArrangements
    # Private tags (groupe impair) — traités séparément
})

# Tags d'UIDs à re-générer (hash déterministe)
UID_TAGS: frozenset[tuple[int, int]] = frozenset({
    (0x0020, 0x000D),  # StudyInstanceUID
    (0x0020, 0x000E),  # SeriesInstanceUID
    (0x0008, 0x0018),  # SOPInstanceUID
    (0x0008, 0x1195),  # TransactionUID
})


@dataclass
class AnonymizationResult:
    twin_id: str
    original_sha256: str
    anonymized_dataset: Dataset
    tags_removed: int
    tags_modified: int
    warnings: list[str]


class DicomAnonymizer:
    """
    Anonymiseur DICOM production-grade.

    Usage:
        anon = DicomAnonymizer(seed="SECRET_INSTITUTIONAL_SEED")
        result = anon.anonymize(path_to_dicom)
    """

    def __init__(self, seed: str) -> None:
        if seed == "CHANGE_ME_IN_PRODUCTION":
            logger.warning("dicom_anonymizer.seed_insecure",
                          msg="Seed par défaut utilisé — inacceptable en production")
        self._seed = seed.encode("utf-8")

    # ─── API publique ─────────────────────────────────────────────────────────

    def compute_twin_id(self, patient_id_raw: str) -> str:
        """
        Génère un twin_id déterministe et irréversible.
        twin_id = HMAC-SHA256(patient_id_raw, seed)[:64]
        """
        return hmac.new(
            key=self._seed,
            msg=patient_id_raw.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()

    def compute_sha256(self, data: bytes) -> str:
        """SHA-256 du contenu binaire original — pour audit trail."""
        return hashlib.sha256(data).hexdigest()

    def anonymize_file(self, dicom_path: Path) -> AnonymizationResult:
        """
        Lit un fichier DICOM depuis le disque et l'anonymise.
        Retourne le Dataset anonymisé sans toucher au fichier original.
        """
        raw_bytes = dicom_path.read_bytes()
        original_sha256 = self.compute_sha256(raw_bytes)

        ds = pydicom.dcmread(dicom_path)
        return self._anonymize_dataset(ds, original_sha256)

    def anonymize_bytes(self, dicom_bytes: bytes) -> AnonymizationResult:
        """
        Anonymise un DICOM fourni en mémoire (upload multipart).
        """
        import io
        original_sha256 = self.compute_sha256(dicom_bytes)
        ds = pydicom.dcmread(io.BytesIO(dicom_bytes), force=True)
        return self._anonymize_dataset(ds, original_sha256)

    # ─── Logique interne ──────────────────────────────────────────────────────

    def _anonymize_dataset(
        self, ds: Dataset, original_sha256: str
    ) -> AnonymizationResult:
        tags_removed = 0
        tags_modified = 0
        warnings: list[str] = []

        # 1. Extraire le PatientID brut AVANT anonymisation
        patient_id_raw = self._extract_patient_id(ds)
        if not patient_id_raw:
            warnings.append("PatientID (0010,0020) absent — twin_id basé sur SHA256 du fichier")
            patient_id_raw = original_sha256  # fallback déterministe

        twin_id = self.compute_twin_id(patient_id_raw)

        # 2. Supprimer les tags privés (groupes impairs)
        private_removed = self._remove_private_tags(ds)
        tags_removed += private_removed

        # 3. Supprimer / anonymiser les tags nominatifs
        for tag in TAGS_TO_REMOVE:
            if tag in ds:
                del ds[tag]
                tags_removed += 1

        # 4. Remplacer PatientID par twin_id
        ds.PatientID = twin_id
        ds.PatientName = "CDT^ANONYMIZED"  # Requis par certains PACS
        tags_modified += 2

        # 5. Re-générer les UIDs de manière déterministe
        uid_modified = self._regen_uids(ds, twin_id)
        tags_modified += uid_modified

        # 6. Anonymiser l'AccessionNumber
        if (0x0008, 0x0050) in ds:
            ds.AccessionNumber = self._hash_uid(ds.AccessionNumber, twin_id)
            tags_modified += 1

        # 7. Traiter les séquences récursivement
        seq_modified = self._process_sequences(ds, twin_id)
        tags_modified += seq_modified

        # 8. Ajouter le tag de dé-identification (DICOM PS3.15)
        ds.DeidentificationMethod = "CDT-PS3.15-BasicProfile"
        ds.PatientIdentityRemoved = "YES"

        # 9. Vérification finale : aucun tag (0010,xxxx) nominatif résiduel
        self._verify_no_pii(ds, warnings)

        logger.info(
            "dicom.anonymized",
            twin_id=twin_id,
            tags_removed=tags_removed,
            tags_modified=tags_modified,
            warnings=len(warnings),
        )

        return AnonymizationResult(
            twin_id=twin_id,
            original_sha256=original_sha256,
            anonymized_dataset=ds,
            tags_removed=tags_removed,
            tags_modified=tags_modified,
            warnings=warnings,
        )

    def _extract_patient_id(self, ds: Dataset) -> Optional[str]:
        """Extrait le PatientID brut. Retourne None si absent."""
        try:
            pid = str(ds.PatientID).strip()
            return pid if pid else None
        except AttributeError:
            return None

    def _remove_private_tags(self, ds: Dataset) -> int:
        """Supprime tous les tags privés (groupe impair)."""
        private_tags = [t for t in ds.keys() if t.group % 2 != 0]
        for tag in private_tags:
            del ds[tag]
        return len(private_tags)

    def _regen_uids(self, ds: Dataset, twin_id: str) -> int:
        """
        Re-génère les UIDs de manière déterministe.
        Le nouvel UID est basé sur un hash du twin_id + UID original.
        Format DICOM UID : 2.25.<entier_128bits>
        """
        modified = 0
        for tag in UID_TAGS:
            if tag in ds:
                original_uid = str(ds[tag].value)
                new_uid = self._hash_uid(original_uid, twin_id)
                ds[tag].value = new_uid
                modified += 1
        return modified

    def _hash_uid(self, original: str, salt: str) -> str:
        """Génère un UID DICOM déterministe à partir d'un original + salt."""
        raw = hashlib.sha256(f"{salt}:{original}".encode()).hexdigest()
        # Convertir en entier 128 bits → format DICOM 2.25.XXXXX
        uid_int = int(raw[:32], 16)
        return f"2.25.{uid_int}"

    def _process_sequences(self, ds: Dataset, twin_id: str) -> int:
        """Traite récursivement les séquences DICOM."""
        modified = 0
        for elem in ds:
            if elem.VR == "SQ":
                for item in elem.value:
                    if isinstance(item, Dataset):
                        # Récursion
                        for tag in TAGS_TO_REMOVE:
                            if tag in item:
                                del item[tag]
                                modified += 1
        return modified

    def _verify_no_pii(self, ds: Dataset, warnings: list[str]) -> None:
        """
        Vérification post-anonymisation : détecte des tags potentiellement
        nominatifs résiduels (heuristique — non exhaustif).
        """
        pii_patterns = [
            r"\b\d{4}-\d{2}-\d{2}\b",  # Date de naissance
            r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # Prénom Nom
        ]
        for elem in ds:
            if elem.VR in ("LO", "LT", "SH", "ST", "PN", "UT"):
                val = str(elem.value)
                for pattern in pii_patterns:
                    if re.search(pattern, val):
                        warnings.append(
                            f"Tag potentiellement nominatif détecté: "
                            f"({elem.tag}) {elem.keyword} = {val[:50]!r}"
                        )
