import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';

/// Screen 3: Grad-CAM View 🔬
///
/// Shows:
/// - Original X-ray image
/// - Grad-CAM heatmap overlay (where the AI is looking)
/// - Toggle between original and heatmap
/// - Explanation of what Grad-CAM means
class GradcamScreen extends StatefulWidget {
  final File imageFile;
  final Uint8List? gradcamBytes;
  final String diagnosis;

  const GradcamScreen({
    super.key,
    required this.imageFile,
    this.gradcamBytes,
    required this.diagnosis,
  });

  @override
  State<GradcamScreen> createState() => _GradcamScreenState();
}

class _GradcamScreenState extends State<GradcamScreen> {
  bool _showHeatmap = true;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('🔬 Grad-CAM Visualization'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Title
            const Text(
              'Model Attention Map',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              'Red/warm areas = regions that influenced the AI\'s "${widget.diagnosis}" decision',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.grey[400], fontSize: 12),
            ),
            const SizedBox(height: 20),

            // Toggle
            Container(
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.05),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: GestureDetector(
                      onTap: () => setState(() => _showHeatmap = false),
                      child: Container(
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        decoration: BoxDecoration(
                          color: !_showHeatmap ? const Color(0xFF4DD0E1).withOpacity(0.2) : null,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(
                          '🩻 Original',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontWeight: !_showHeatmap ? FontWeight.bold : FontWeight.normal,
                            color: !_showHeatmap ? const Color(0xFF4DD0E1) : Colors.grey,
                          ),
                        ),
                      ),
                    ),
                  ),
                  Expanded(
                    child: GestureDetector(
                      onTap: () => setState(() => _showHeatmap = true),
                      child: Container(
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        decoration: BoxDecoration(
                          color: _showHeatmap ? const Color(0xFF7C4DFF).withOpacity(0.2) : null,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(
                          '🔥 Heatmap',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontWeight: _showHeatmap ? FontWeight.bold : FontWeight.normal,
                            color: _showHeatmap ? const Color(0xFF7C4DFF) : Colors.grey,
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // Image Display
            ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: AnimatedSwitcher(
                duration: const Duration(milliseconds: 300),
                child: _showHeatmap && widget.gradcamBytes != null
                    ? Image.memory(
                        widget.gradcamBytes!,
                        key: const ValueKey('heatmap'),
                        fit: BoxFit.contain,
                      )
                    : Image.file(
                        widget.imageFile,
                        key: const ValueKey('original'),
                        fit: BoxFit.contain,
                      ),
              ),
            ),
            const SizedBox(height: 24),

            // Explanation
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.03),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.white.withOpacity(0.08)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'ℹ️ What is Grad-CAM?',
                    style: TextStyle(fontWeight: FontWeight.w600, fontSize: 14),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Gradient-weighted Class Activation Mapping (Grad-CAM) visualizes which '
                    'regions of the X-ray the deep learning model focused on to make its prediction.\n\n'
                    '• Red/warm areas = high attention (most influential)\n'
                    '• Blue/cool areas = low attention\n\n'
                    'This helps medical professionals understand and verify the AI\'s reasoning.',
                    style: TextStyle(color: Colors.grey[400], fontSize: 12, height: 1.5),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
