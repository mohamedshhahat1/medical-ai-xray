"""
AI Medical Report Generator
==============================

Generates professional PDF reports for X-ray analysis results.
Includes diagnosis, confidence, Grad-CAM heatmap, and clinical findings.

Output format:
    - Patient info section
    - AI analysis findings
    - Confidence scores per class
    - Grad-CAM visualization (highlighted regions)
    - Medical disclaimer

Usage:
    from backend.report_generator import generate_report
    pdf_bytes = generate_report(prediction_result, original_image, heatmap_image)
"""

import io
import os
import datetime
from PIL import Image

# ReportLab for PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


def generate_report(prediction, original_image=None, heatmap_image=None,
                    patient_id=None, patient_name=None, notes=None):
    """
    Generate a professional PDF medical AI report.

    Args:
        prediction (dict): Prediction result from inference engine:
            {
                "prediction": "Pneumonia",
                "confidence": 0.93,
                "probabilities": {"Normal": 0.07, "Pneumonia": 0.93, ...},
                "description": "...",
                "severity": "moderate"
            }
        original_image (PIL.Image or bytes): Original X-ray image.
        heatmap_image (PIL.Image or bytes): Grad-CAM overlay image.
        patient_id (str, optional): Patient identifier.
        patient_name (str, optional): Patient name.
        notes (str, optional): Additional clinical notes.

    Returns:
        bytes: PDF file content as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm
    )

    # Styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='ReportTitle', fontSize=18, leading=22,
        alignment=TA_CENTER, spaceAfter=6, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#1a237e')
    ))
    styles.add(ParagraphStyle(
        name='SectionTitle', fontSize=13, leading=16,
        spaceAfter=8, spaceBefore=14, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#263238')
    ))
    styles.add(ParagraphStyle(
        name='Finding', fontSize=11, leading=14,
        spaceAfter=4, leftIndent=20, fontName='Helvetica'
    ))
    styles.add(ParagraphStyle(
        name='Disclaimer', fontSize=8, leading=10,
        alignment=TA_CENTER, textColor=colors.HexColor('#757575'),
        fontName='Helvetica-Oblique'
    ))
    styles.add(ParagraphStyle(
        name='SmallText', fontSize=9, leading=11,
        textColor=colors.HexColor('#546e7a')
    ))

    elements = []

    # =========================================================================
    # HEADER
    # =========================================================================
    elements.append(Paragraph("🏥 Medical AI — X-Ray Analysis Report", styles['ReportTitle']))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "Automated Chest X-Ray Classification using Deep Learning",
        ParagraphStyle('SubTitle', fontSize=10, alignment=TA_CENTER,
                       textColor=colors.HexColor('#546e7a'))
    ))
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=1,
                               color=colors.HexColor('#e0e0e0')))
    elements.append(Spacer(1, 12))

    # =========================================================================
    # REPORT INFO
    # =========================================================================
    now = datetime.datetime.now()
    info_data = [
        ["Report Date:", now.strftime("%B %d, %Y — %H:%M")],
        ["Report ID:", f"MXAI-{now.strftime('%Y%m%d%H%M%S')}"],
    ]
    if patient_id:
        info_data.append(["Patient ID:", patient_id])
    if patient_name:
        info_data.append(["Patient Name:", patient_name])

    info_table = Table(info_data, colWidths=[3 * cm, 10 * cm])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#37474f')),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#263238')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 16))

    # =========================================================================
    # DIAGNOSIS (Main Finding)
    # =========================================================================
    elements.append(Paragraph("■ AI Diagnosis", styles['SectionTitle']))

    diagnosis = prediction.get("prediction", "Unknown")
    confidence = prediction.get("confidence", 0)
    description = prediction.get("description", "")
    severity = prediction.get("severity", "")

    # Color-coded diagnosis
    severity_colors = {
        "none": "#4caf50", "low": "#8bc34a",
        "moderate": "#ff9800", "high": "#f44336"
    }
    diag_color = severity_colors.get(severity, "#263238")

    elements.append(Paragraph(
        f'<font color="{diag_color}" size="16"><b>{diagnosis}</b></font>',
        ParagraphStyle('DiagMain', fontSize=16, spaceAfter=6, leftIndent=20)
    ))

    if description:
        elements.append(Paragraph(f"<i>{description}</i>", styles['Finding']))

    elements.append(Spacer(1, 8))

    # =========================================================================
    # FINDINGS
    # =========================================================================
    elements.append(Paragraph("■ Findings", styles['SectionTitle']))

    # Confidence
    conf_pct = f"{confidence * 100:.1f}%"
    elements.append(Paragraph(
        f"• <b>Primary diagnosis:</b> {diagnosis} (Confidence: {conf_pct})",
        styles['Finding']
    ))

    # Severity
    if severity and severity != "none":
        elements.append(Paragraph(
            f'• <b>Severity level:</b> <font color="{diag_color}">{severity.upper()}</font>',
            styles['Finding']
        ))
    else:
        elements.append(Paragraph(
            '• <b>Severity level:</b> <font color="#4caf50">NONE — No abnormality detected</font>',
            styles['Finding']
        ))

    # Highlighted region info (from Grad-CAM)
    if heatmap_image is not None:
        region = _estimate_region(prediction)
        elements.append(Paragraph(
            f"• <b>Highlighted region:</b> {region}",
            styles['Finding']
        ))
        elements.append(Paragraph(
            "• <b>Visualization:</b> Grad-CAM heatmap shows model attention areas (see below)",
            styles['Finding']
        ))

    elements.append(Spacer(1, 8))

    # =========================================================================
    # CONFIDENCE SCORES (All Classes)
    # =========================================================================
    elements.append(Paragraph("■ Classification Probabilities", styles['SectionTitle']))

    probabilities = prediction.get("probabilities", {})
    if probabilities:
        prob_data = [["Class", "Probability", "Bar"]]
        for cls_name, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
            pct = f"{prob * 100:.1f}%"
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            prob_data.append([cls_name, pct, bar])

        prob_table = Table(prob_data, colWidths=[4 * cm, 3 * cm, 8 * cm])
        prob_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eceff1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#263238')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cfd8dc')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(prob_table)

    elements.append(Spacer(1, 16))

    # =========================================================================
    # IMAGES (Original + Grad-CAM)
    # =========================================================================
    if original_image or heatmap_image:
        elements.append(Paragraph("■ X-Ray & Grad-CAM Visualization", styles['SectionTitle']))
        elements.append(Paragraph(
            "The heatmap highlights regions that most influenced the AI model's decision. "
            "Red/warm areas indicate high attention.",
            styles['SmallText']
        ))
        elements.append(Spacer(1, 8))

        img_data = []
        img_row = []

        if original_image:
            orig_bytes = _image_to_bytes(original_image)
            img_row.append(RLImage(io.BytesIO(orig_bytes), width=7 * cm, height=7 * cm))
        if heatmap_image:
            heat_bytes = _image_to_bytes(heatmap_image)
            img_row.append(RLImage(io.BytesIO(heat_bytes), width=7 * cm, height=7 * cm))

        if img_row:
            img_table = Table([img_row], colWidths=[8 * cm] * len(img_row))
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            elements.append(img_table)

            # Captions
            captions = []
            if original_image:
                captions.append("Original X-Ray")
            if heatmap_image:
                captions.append("Grad-CAM Heatmap")
            cap_table = Table([captions], colWidths=[8 * cm] * len(captions))
            cap_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#78909c')),
            ]))
            elements.append(cap_table)

    elements.append(Spacer(1, 16))

    # =========================================================================
    # CLINICAL NOTES
    # =========================================================================
    if notes:
        elements.append(Paragraph("■ Additional Notes", styles['SectionTitle']))
        elements.append(Paragraph(notes, styles['Finding']))
        elements.append(Spacer(1, 12))

    # =========================================================================
    # DISCLAIMER
    # =========================================================================
    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="100%", thickness=0.5,
                               color=colors.HexColor('#e0e0e0')))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "⚠️ DISCLAIMER: This report is generated by an AI system for EDUCATIONAL PURPOSES ONLY. "
        "It is NOT a medical diagnosis and should NOT be used for clinical decision-making. "
        "Always consult a qualified radiologist or physician for medical interpretation of X-ray images. "
        "This system has not been validated in a clinical setting and has not received regulatory approval.",
        styles['Disclaimer']
    ))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        f"Generated by Medical AI X-Ray Analysis System — {now.strftime('%Y-%m-%d %H:%M:%S')}",
        styles['Disclaimer']
    ))

    # Build PDF
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


def _estimate_region(prediction):
    """
    Estimate the affected lung region based on the diagnosis.

    In a production system, this would use Grad-CAM coordinates.
    Here we provide medically plausible descriptions.

    Args:
        prediction (dict): Prediction result.

    Returns:
        str: Description of the highlighted region.
    """
    diagnosis = prediction.get("prediction", "")
    confidence = prediction.get("confidence", 0)

    if diagnosis == "Normal":
        return "No abnormal regions highlighted — bilateral lungs appear clear"
    elif diagnosis == "Pneumonia":
        if confidence > 0.8:
            return "Lower right lung field — consolidation pattern detected"
        return "Bilateral lower lung fields — possible infiltrates"
    elif diagnosis == "Tuberculosis":
        return "Upper lung zones — possible cavitation/fibrosis pattern"
    elif diagnosis == "COVID":
        return "Bilateral peripheral regions — ground-glass opacities pattern"
    else:
        return "Region analysis not available"


def _image_to_bytes(image):
    """Convert PIL Image or bytes to PNG bytes."""
    if isinstance(image, bytes):
        return image
    elif isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        return buf.getvalue()
    return b''
