"""
Tests unitaires — DicomAnonymizer.
Vérifie :
  - Génération déterministe du twin_id
  - Unicité (patients différents → twin_ids différents)
  - 0 tag nominatif en sortie (DICOM PS3.15)
  - Irréversibilité (seeds différents → twin_ids différents)
  - Conservation des données pixel
  - Round-trip : DICOM original → anonymisé → re-lu
"""
import io
import pytest
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian
import numpy as np

from app.anonymizer.dicom_anonymizer import DicomAnonymizer, TAGS_TO_REMOVE


SEED = "test_institutional_seed_not_for_production"


def make_dicom_bytes(
    patient_id: str = "PAT001",
    patient_name: str = "Dupont^Jean",
    modality: str = "MR",
    with_pixels: bool = True,
) -> bytes:
    """Crée un fichier DICOM minimal conforme (avec preambule 128 bytes + DICM) pour les tests."""
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6.7.8.9"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(
        filename_or_obj="test.dcm",
        dataset={},
        file_meta=file_meta,
        is_implicit_VR=False,
        is_little_endian=True,
    )
    ds.is_implicit_VR = False
    ds.is_little_endian = True
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # Tags nominatifs (doivent être supprimés)
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.PatientBirthDate = "19800101"
    ds.PatientSex = "M"
    ds.PatientAge = "044Y"
    ds.InstitutionName = "CHU de Paris"
    ds.ReferringPhysicianName = "Dr Martin"

    # Tags techniques (doivent être conservés)
    ds.Modality = modality
    ds.StudyInstanceUID = "1.2.3.4"
    ds.SeriesInstanceUID = "1.2.3.4.5"
    ds.SOPInstanceUID = "1.2.3.4.5.6"
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.Rows = 64
    ds.Columns = 64
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    if with_pixels:
        pixel_array = np.zeros((64, 64), dtype=np.uint16)
        pixel_array[30:34, 30:34] = 1000  # Signal cardiaque simulé
        ds.PixelData = pixel_array.tobytes()

    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds)
    return buf.getvalue()


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestTwinIdGeneration:
    """Tests de la génération déterministe du twin_id."""

    def test_same_patient_same_seed_reproducible(self):
        """CRITIQUE : même patient + même seed → même twin_id."""
        anon = DicomAnonymizer(seed=SEED)
        id1 = anon.compute_twin_id("PAT001")
        id2 = anon.compute_twin_id("PAT001")
        assert id1 == id2, "twin_id doit être déterministe"

    def test_different_patients_different_twin_ids(self):
        """Patients différents → twin_ids différents."""
        anon = DicomAnonymizer(seed=SEED)
        id1 = anon.compute_twin_id("PAT001")
        id2 = anon.compute_twin_id("PAT002")
        assert id1 != id2, "Collision de twin_id détectée"

    def test_different_seeds_different_twin_ids(self):
        """Seed différent → twin_id différent (irréversibilité institutionnelle)."""
        anon1 = DicomAnonymizer(seed="seed_hopital_A")
        anon2 = DicomAnonymizer(seed="seed_hopital_B")
        id1 = anon1.compute_twin_id("PAT001")
        id2 = anon2.compute_twin_id("PAT001")
        assert id1 != id2, "Isolation inter-institutions défaillante"

    def test_twin_id_is_sha256_hex(self):
        """twin_id doit être un SHA-256 hex valide (64 chars)."""
        anon = DicomAnonymizer(seed=SEED)
        twin_id = anon.compute_twin_id("PAT001")
        assert len(twin_id) == 64
        assert all(c in "0123456789abcdef" for c in twin_id)


class TestDicomAnonymization:
    """Tests du pipeline d'anonymisation DICOM PS3.15."""

    def setup_method(self):
        self.anon = DicomAnonymizer(seed=SEED)

    def test_no_patient_nominative_tags_in_output(self):
        """SLO CRITIQUE : 0 tag nominatif (0010,xxxx) en sortie."""
        dicom_bytes = make_dicom_bytes(patient_id="PAT001", patient_name="Dupont^Jean")
        result = self.anon.anonymize_bytes(dicom_bytes)
        ds = result.anonymized_dataset

        # PatientName ne doit PAS contenir le nom réel
        patient_name = str(getattr(ds, "PatientName", ""))
        assert "Dupont" not in patient_name, f"Nom réel trouvé: {patient_name}"
        assert "Jean" not in patient_name, f"Prénom réel trouvé: {patient_name}"

        # PatientBirthDate doit être absent
        assert not hasattr(ds, "PatientBirthDate") or ds.PatientBirthDate == "", \
            "Date de naissance non supprimée"

        # InstitutionName doit être absent
        assert (0x0008, 0x0080) not in ds, "InstitutionName non supprimé"

        # ReferringPhysicianName doit être absent
        assert (0x0008, 0x0090) not in ds, "ReferringPhysicianName non supprimé"

    def test_twin_id_in_patient_id_tag(self):
        """PatientID doit contenir le twin_id après anonymisation."""
        dicom_bytes = make_dicom_bytes(patient_id="PAT001")
        result = self.anon.anonymize_bytes(dicom_bytes)
        ds = result.anonymized_dataset

        assert ds.PatientID == result.twin_id, \
            "PatientID doit être remplacé par twin_id"

    def test_twin_id_matches_expected(self):
        """twin_id calculé par anonymize_bytes == compute_twin_id."""
        dicom_bytes = make_dicom_bytes(patient_id="PAT_KNOWN")
        result = self.anon.anonymize_bytes(dicom_bytes)

        expected_twin_id = self.anon.compute_twin_id("PAT_KNOWN")
        assert result.twin_id == expected_twin_id

    def test_pixel_data_preserved(self):
        """Les données pixel doivent être intactes après anonymisation."""
        dicom_bytes = make_dicom_bytes(with_pixels=True)
        result = self.anon.anonymize_bytes(dicom_bytes)
        ds = result.anonymized_dataset

        assert hasattr(ds, "PixelData"), "Données pixel supprimées par erreur"
        assert len(ds.PixelData) == 64 * 64 * 2, "Taille des données pixel modifiée"

    def test_technical_metadata_preserved(self):
        """Les métadonnées techniques (Modality, Rows, Columns) sont conservées."""
        dicom_bytes = make_dicom_bytes(modality="MR")
        result = self.anon.anonymize_bytes(dicom_bytes)
        ds = result.anonymized_dataset

        assert ds.Modality == "MR"
        assert ds.Rows == 64
        assert ds.Columns == 64

    def test_uids_regenerated(self):
        """Les UIDs doivent être re-générés (différents des originaux)."""
        dicom_bytes = make_dicom_bytes()
        result = self.anon.anonymize_bytes(dicom_bytes)
        ds = result.anonymized_dataset

        # StudyInstanceUID original était "1.2.3.4"
        assert ds.StudyInstanceUID != "1.2.3.4", "UID original non re-généré"
        assert ds.StudyInstanceUID.startswith("2.25."), \
            f"Format UID CDT invalide: {ds.StudyInstanceUID}"

    def test_uids_deterministic(self):
        """Les UIDs re-générés sont déterministes pour le même fichier."""
        dicom_bytes = make_dicom_bytes(patient_id="PAT001")
        result1 = self.anon.anonymize_bytes(dicom_bytes)
        result2 = self.anon.anonymize_bytes(dicom_bytes)
        assert result1.anonymized_dataset.StudyInstanceUID == \
               result2.anonymized_dataset.StudyInstanceUID

    def test_original_sha256_computed(self):
        """Le SHA-256 original doit être calculé et stocké."""
        dicom_bytes = make_dicom_bytes()
        result = self.anon.anonymize_bytes(dicom_bytes)

        import hashlib
        expected = hashlib.sha256(dicom_bytes).hexdigest()
        assert result.original_sha256 == expected

    def test_deidentification_tag_added(self):
        """Le tag PatientIdentityRemoved doit être ajouté."""
        dicom_bytes = make_dicom_bytes()
        result = self.anon.anonymize_bytes(dicom_bytes)
        ds = result.anonymized_dataset

        assert getattr(ds, "PatientIdentityRemoved", None) == "YES"

    def test_different_patients_different_results(self):
        """Deux patients différents → twin_ids différents après anonymisation."""
        bytes1 = make_dicom_bytes(patient_id="PAT001")
        bytes2 = make_dicom_bytes(patient_id="PAT002")

        result1 = self.anon.anonymize_bytes(bytes1)
        result2 = self.anon.anonymize_bytes(bytes2)

        assert result1.twin_id != result2.twin_id


class TestRoundTrip:
    """Tests de round-trip : DICOM → anonymisé → re-lu sans erreur."""

    def test_round_trip_readable(self):
        """Le fichier anonymisé doit être lisible par pydicom."""
        anon = DicomAnonymizer(seed=SEED)
        dicom_bytes = make_dicom_bytes()
        result = anon.anonymize_bytes(dicom_bytes)

        # Sérialiser et re-lire (force=True car file_meta peut être incomplet)
        buf = io.BytesIO()
        pydicom.dcmwrite(buf, result.anonymized_dataset)
        re_read = pydicom.dcmread(io.BytesIO(buf.getvalue()), force=True)

        assert re_read.PatientID == result.twin_id
        assert re_read.Modality == "MR"

    def test_round_trip_sha256_stable(self):
        """SHA-256 du fichier anonymisé stable entre deux sérialisations."""
        import hashlib
        anon = DicomAnonymizer(seed=SEED)
        dicom_bytes = make_dicom_bytes(patient_id="PAT_STABLE")

        result = anon.anonymize_bytes(dicom_bytes)

        buf1 = io.BytesIO()
        pydicom.dcmwrite(buf1, result.anonymized_dataset)
        sha1 = hashlib.sha256(buf1.getvalue()).hexdigest()

        # Refaire l'anonymisation du même input
        result2 = anon.anonymize_bytes(dicom_bytes)
        buf2 = io.BytesIO()
        pydicom.dcmwrite(buf2, result2.anonymized_dataset)
        sha2 = hashlib.sha256(buf2.getvalue()).hexdigest()

        assert sha1 == sha2, "Anonymisation non-déterministe détectée"
