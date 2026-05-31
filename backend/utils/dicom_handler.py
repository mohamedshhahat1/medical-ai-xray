"""
DICOM Support for Medical X-Ray Images
=========================================

Handles loading, parsing, and preprocessing of DICOM files (.dcm),
the standard format used in real hospitals and radiology departments.

DICOM (Digital Imaging and Communications in Medicine):
    - Standard format for ALL medical imaging worldwide
    - Contains image data + patient metadata (name, age, study info)
    - Supports various bit depths (8-bit, 12-bit, 16-bit)
    - May include windowing parameters (window center/width)
    - Compressed (JPEG2000, RLE) or uncompressed

Why DICOM matters:
    Real hospital PACS (Picture Archiving and Communication System)
    stores ALL X-rays as DICOM. If your AI can't read DICOM, it
    can't work in a real clinical environment.

Features:
    - Load .dcm files and convert to PIL Image
    - Extract patient metadata (anonymizable)
    - Apply proper windowing (contrast adjustment)
    - Handle multiple transfer syntaxes (compression)
    - DICOM → PNG/JPG conversion utility
    - Batch processing for entire study folders

Usage:
    from backend.utils.dicom_handler import load_dicom, dicom_to_pil, get_metadata

    # Load and convert
    image = load_dicom("scan.dcm")

    # Get patient info
    metadata = get_metadata("scan.dcm")

    # Batch convert
    convert_dicom_folder("/hospital/study_001/", output_dir="./converted/")
"""

import os
import io
import numpy as np
from PIL import Image

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


# =============================================================================
# DICOM LOADING
# =============================================================================

def load_dicom(filepath):
    """
    Load a DICOM file and return as a normalized numpy array.

    Handles:
    - Various bit depths (8, 12, 16 bit)
    - Photometric interpretation (MONOCHROME1/2)
    - VOI LUT (Value of Interest Lookup Table) windowing
    - Rescale slope/intercept (Hounsfield units for CT)

    Args:
        filepath (str): Path to .dcm file.

    Returns:
        np.ndarray: Normalized image array (0-255, uint8).

    Raises:
        ImportError: If pydicom is not installed.
        FileNotFoundError: If file doesn't exist.
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError(
            "pydicom is required for DICOM support.\n"
            "Install with: pip install pydicom pylibjpeg pylibjpeg-libjpeg"
        )

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"DICOM file not found: {filepath}")

    # Read DICOM file
    ds = pydicom.dcmread(filepath)

    # Get pixel data
    pixel_array = ds.pixel_array.astype(np.float32)

    # Apply VOI LUT (windowing) if available
    try:
        pixel_array = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
    except Exception:
        pass

    # Apply rescale slope/intercept if present (common in CT, sometimes in CR)
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

    # Handle photometric interpretation
    # MONOCHROME1: air is white, bone is black (inverted)
    # MONOCHROME2: air is black, bone is white (standard)
    if hasattr(ds, 'PhotometricInterpretation'):
        if ds.PhotometricInterpretation == "MONOCHROME1":
            pixel_array = pixel_array.max() - pixel_array

    # Normalize to 0-255
    if pixel_array.max() > pixel_array.min():
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
    pixel_array = (pixel_array * 255).astype(np.uint8)

    return pixel_array


def dicom_to_pil(filepath, apply_windowing=True):
    """
    Load a DICOM file and convert to PIL Image.

    Args:
        filepath (str): Path to .dcm file.
        apply_windowing (bool): Whether to apply DICOM windowing.

    Returns:
        PIL.Image: Grayscale image ready for model input.
    """
    pixel_array = load_dicom(filepath)
    return Image.fromarray(pixel_array, mode='L')


def dicom_to_bytes(filepath):
    """
    Load a DICOM file and return as PNG bytes (for API compatibility).

    Args:
        filepath (str): Path to .dcm file.

    Returns:
        bytes: PNG-encoded image bytes.
    """
    image = dicom_to_pil(filepath)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


# =============================================================================
# DICOM WINDOWING (Manual Control)
# =============================================================================

def apply_window(pixel_array, window_center, window_width):
    """
    Apply manual windowing (contrast adjustment) to DICOM pixel data.

    Windowing controls how the raw pixel values map to display brightness.
    Different windows reveal different structures:
        - Lung window: center=-600, width=1500 (shows lung detail)
        - Bone window: center=300, width=1500 (shows bone detail)
        - Soft tissue: center=40, width=400 (shows soft tissue)

    For chest X-rays (CR/DR), common window:
        - center=2048, width=4096 (full range)

    Args:
        pixel_array (np.ndarray): Raw pixel data.
        window_center (float): Center of the display window.
        window_width (float): Width of the display window.

    Returns:
        np.ndarray: Windowed image normalized to 0-255.
    """
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2

    windowed = np.clip(pixel_array, img_min, img_max)
    windowed = ((windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)

    return windowed


# =============================================================================
# METADATA EXTRACTION
# =============================================================================

def get_metadata(filepath, anonymize=True):
    """
    Extract metadata from a DICOM file.

    Returns patient info, study info, and technical parameters.
    Can anonymize (remove patient-identifying information) for privacy.

    Args:
        filepath (str): Path to .dcm file.
        anonymize (bool): If True, removes patient name/ID (HIPAA compliance).

    Returns:
        dict: Extracted metadata fields.
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom required. Install: pip install pydicom")

    ds = pydicom.dcmread(filepath)

    metadata = {
        # Study information
        "study_date": str(getattr(ds, 'StudyDate', '')),
        "study_description": str(getattr(ds, 'StudyDescription', '')),
        "modality": str(getattr(ds, 'Modality', '')),
        "body_part": str(getattr(ds, 'BodyPartExamined', '')),
        "view_position": str(getattr(ds, 'ViewPosition', '')),

        # Image parameters
        "rows": int(getattr(ds, 'Rows', 0)),
        "columns": int(getattr(ds, 'Columns', 0)),
        "bits_allocated": int(getattr(ds, 'BitsAllocated', 0)),
        "bits_stored": int(getattr(ds, 'BitsStored', 0)),
        "photometric": str(getattr(ds, 'PhotometricInterpretation', '')),
        "pixel_spacing": list(getattr(ds, 'PixelSpacing', [])),

        # Equipment
        "manufacturer": str(getattr(ds, 'Manufacturer', '')),
        "institution": str(getattr(ds, 'InstitutionName', '')),
        "station_name": str(getattr(ds, 'StationName', '')),

        # Window settings
        "window_center": _get_window_value(ds, 'WindowCenter'),
        "window_width": _get_window_value(ds, 'WindowWidth'),
    }

    # Patient info (conditionally included)
    if not anonymize:
        metadata.update({
            "patient_name": str(getattr(ds, 'PatientName', '')),
            "patient_id": str(getattr(ds, 'PatientID', '')),
            "patient_age": str(getattr(ds, 'PatientAge', '')),
            "patient_sex": str(getattr(ds, 'PatientSex', '')),
        })
    else:
        metadata.update({
            "patient_name": "[ANONYMIZED]",
            "patient_id": "[ANONYMIZED]",
            "patient_age": str(getattr(ds, 'PatientAge', '')),
            "patient_sex": str(getattr(ds, 'PatientSex', '')),
        })

    return metadata


def _get_window_value(ds, attr_name):
    """Safely extract window center/width (may be a list or single value)."""
    val = getattr(ds, attr_name, None)
    if val is None:
        return None
    if isinstance(val, (list, pydicom.multival.MultiValue)):
        return float(val[0])
    return float(val)


# =============================================================================
# BATCH CONVERSION
# =============================================================================

def convert_dicom_folder(input_dir, output_dir, output_format='png'):
    """
    Batch convert all DICOM files in a folder to PNG/JPG.

    Useful for preparing hospital data for training/validation.

    Args:
        input_dir (str): Folder containing .dcm files (recursive search).
        output_dir (str): Output folder for converted images.
        output_format (str): 'png' or 'jpg'.

    Returns:
        int: Number of files converted.
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom required. Install: pip install pydicom")

    os.makedirs(output_dir, exist_ok=True)
    converted = 0
    errors = 0

    # Find all DICOM files (recursive)
    dcm_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.dcm', '.dicom')) or '.' not in f:
                dcm_files.append(os.path.join(root, f))

    print(f"  Found {len(dcm_files)} potential DICOM files")

    for dcm_path in dcm_files:
        try:
            image = dicom_to_pil(dcm_path)

            # Create output filename
            rel_path = os.path.relpath(dcm_path, input_dir)
            out_name = os.path.splitext(rel_path)[0] + f'.{output_format}'
            out_path = os.path.join(output_dir, out_name)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            image.save(out_path)
            converted += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ⚠️  Failed: {os.path.basename(dcm_path)} — {e}")

    print(f"  ✓ Converted: {converted}/{len(dcm_files)} files")
    if errors > 0:
        print(f"  ⚠️  Errors: {errors}")

    return converted


def is_dicom_file(filepath):
    """
    Check if a file is a valid DICOM file.

    Args:
        filepath (str): Path to check.

    Returns:
        bool: True if valid DICOM.
    """
    if not PYDICOM_AVAILABLE:
        return filepath.lower().endswith(('.dcm', '.dicom'))

    try:
        pydicom.dcmread(filepath, stop_before_pixels=True)
        return True
    except Exception:
        return False
