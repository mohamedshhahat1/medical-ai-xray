import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:open_file/open_file.dart';
import 'package:share_plus/share_plus.dart';
import '../services/api_service.dart';
import 'gradcam_screen.dart';

/// Screen 2: Result Screen 🧠
///
/// Shows:
/// - Diagnosis (color-coded by severity)
/// - Confidence percentage (circular indicator)
/// - All class probabilities (bar chart)
/// - Description of the condition
/// - Buttons: View Grad-CAM, Download Report
class ResultScreen extends StatelessWidget {
  final File imageFile;
  final Map<String, dynamic> prediction;
  final Uint8List? gradcamBytes;

  const ResultScreen({
    super.key,
    required this.imageFile,
    required this.prediction,
    this.gradcamBytes,
  });

  Color _getSeverityColor(String severity) {
    switch (severity) {
      case 'high': return Colors.red;
      case 'moderate': return Colors.orange;
      case 'low': return Colors.yellow;
      default: return Colors.green;
    }
  }

  Color _getClassColor(String className) {
    switch (className.toLowerCase()) {
      case 'normal': return Colors.green;
      case 'pneumonia': return Colors.orange;
      case 'tuberculosis': return Colors.red;
      case 'covid': return Colors.purple;
      default: return Colors.cyan;
    }
  }

  @override
  Widget build(BuildContext context) {
    final String diagnosis = prediction['prediction'] ?? 'Unknown';
    final double confidence = (prediction['confidence'] ?? 0).toDouble();
    final Map<String, dynamic> probs = Map<String, dynamic>.from(prediction['probabilities'] ?? {});
    final String description = prediction['description'] ?? '';
    final String severity = prediction['severity'] ?? 'none';
    final Color sevColor = _getSeverityColor(severity);

    return Scaffold(
      appBar: AppBar(
        title: const Text('🧠 Analysis Result'),
        actions: [
          IconButton(
            icon: const Icon(Icons.share),
            onPressed: () => _shareResult(context),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Diagnosis Card
            Container(
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [sevColor.withOpacity(0.15), sevColor.withOpacity(0.05)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: sevColor.withOpacity(0.3)),
              ),
              child: Column(
                children: [
                  Text(
                    diagnosis,
                    style: TextStyle(
                      fontSize: 32,
                      fontWeight: FontWeight.bold,
                      color: _getClassColor(diagnosis),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Confidence: ${(confidence * 100).toStringAsFixed(1)}%',
                    style: TextStyle(fontSize: 16, color: Colors.grey[300]),
                  ),
                  if (severity != 'none') ...[
                    const SizedBox(height: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                      decoration: BoxDecoration(
                        color: sevColor.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        'Severity: ${severity.toUpperCase()}',
                        style: TextStyle(color: sevColor, fontSize: 12, fontWeight: FontWeight.w600),
                      ),
                    ),
                  ],
                  if (description.isNotEmpty) ...[
                    const SizedBox(height: 12),
                    Text(
                      description,
                      textAlign: TextAlign.center,
                      style: TextStyle(color: Colors.grey[400], fontSize: 13),
                    ),
                  ],
                ],
              ),
            ),
            const SizedBox(height: 20),

            // Probabilities
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.03),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.white.withOpacity(0.08)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Classification Probabilities',
                    style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: Colors.grey),
                  ),
                  const SizedBox(height: 16),
                  ...probs.entries.toList()
                    ..sort((a, b) => (b.value as num).compareTo(a.value as num))
                    ..map((entry) {
                      final pct = ((entry.value as num) * 100);
                      return Padding(
                        padding: const EdgeInsets.only(bottom: 12),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(entry.key, style: const TextStyle(fontSize: 13)),
                                Text('${pct.toStringAsFixed(1)}%',
                                    style: TextStyle(fontSize: 13, color: Colors.grey[400])),
                              ],
                            ),
                            const SizedBox(height: 4),
                            ClipRRect(
                              borderRadius: BorderRadius.circular(4),
                              child: LinearProgressIndicator(
                                value: pct / 100,
                                backgroundColor: Colors.white.withOpacity(0.05),
                                valueColor: AlwaysStoppedAnimation(_getClassColor(entry.key)),
                                minHeight: 8,
                              ),
                            ),
                          ],
                        ),
                      );
                    }).toList(),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // Action Buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () {
                      Navigator.push(context, MaterialPageRoute(
                        builder: (_) => GradcamScreen(
                          imageFile: imageFile,
                          gradcamBytes: gradcamBytes,
                          diagnosis: diagnosis,
                        ),
                      ));
                    },
                    icon: const Icon(Icons.remove_red_eye),
                    label: const Text('Grad-CAM'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF7C4DFF),
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _downloadReport(context),
                    icon: const Icon(Icons.picture_as_pdf),
                    label: const Text('PDF Report'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF4DD0E1),
                      foregroundColor: Colors.black,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    ),
                  ),
                ),
              ],
            ),

            // Disclaimer
            const SizedBox(height: 24),
            Text(
              '⚠️ This is an AI-generated analysis for educational purposes only.\nNot for clinical diagnosis.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 10, color: Colors.grey[600]),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _downloadReport(BuildContext context) async {
    try {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('⏳ Generating PDF report...')),
      );

      final pdfBytes = await ApiService.getReport(imageFile);

      // Save to device
      final dir = await getApplicationDocumentsDirectory();
      final file = File('${dir.path}/xray_report.pdf');
      await file.writeAsBytes(pdfBytes);

      if (!context.mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: const Text('✅ Report saved!'),
          action: SnackBarAction(
            label: 'OPEN',
            onPressed: () => OpenFile.open(file.path),
          ),
        ),
      );
    } catch (e) {
      if (!context.mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('❌ Error: $e'), backgroundColor: Colors.red),
      );
    }
  }

  void _shareResult(BuildContext context) {
    final text = '🏥 Medical AI X-Ray Analysis\n\n'
        'Diagnosis: ${ prediction['prediction']}\n'
        'Confidence: ${((prediction['confidence'] ?? 0) * 100).toStringAsFixed(1)}%\n'
        'Severity: ${prediction['severity'] ?? 'N/A'}\n\n'
        '⚠️ Educational purposes only — NOT for clinical use.';
    Share.share(text);
  }
}
