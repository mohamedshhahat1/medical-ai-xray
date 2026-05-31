import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/api_service.dart';
import 'result_screen.dart';

/// Screen 1: Upload X-Ray 📷
///
/// Features:
/// - Camera capture
/// - Gallery picker
/// - Drag & drop area (visual)
/// - Image preview before sending
/// - Loading animation while analyzing
class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key});

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen>
    with SingleTickerProviderStateMixin {
  File? _selectedImage;
  bool _isAnalyzing = false;
  late AnimationController _pulseController;
  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  Future<void> _pickImage(ImageSource source) async {
    final XFile? image = await _picker.pickImage(
      source: source,
      maxWidth: 1024,
      maxHeight: 1024,
      imageQuality: 90,
    );

    if (image != null) {
      setState(() => _selectedImage = File(image.path));
      // Auto-analyze
      _analyzeImage();
    }
  }

  Future<void> _analyzeImage() async {
    if (_selectedImage == null) return;

    setState(() => _isAnalyzing = true);

    try {
      // Call API
      final result = await ApiService.predict(_selectedImage!);

      // Get Grad-CAM
      Uint8List? gradcam;
      try {
        gradcam = await ApiService.getGradCAM(_selectedImage!);
      } catch (_) {}

      if (!mounted) return;

      // Navigate to results
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => ResultScreen(
            imageFile: _selectedImage!,
            prediction: result,
            gradcamBytes: gradcam,
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error: ${e.toString()}'),
          backgroundColor: Colors.red,
        ),
      );
    } finally {
      if (mounted) setState(() => _isAnalyzing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('🏥 Medical AI X-Ray'),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Column(
            children: [
              // Header
              const Text(
                'Chest X-Ray Analysis',
                style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              Text(
                'Upload an X-ray for AI-powered diagnosis',
                style: TextStyle(color: Colors.grey[400], fontSize: 14),
              ),
              const SizedBox(height: 8),
              // Disclaimer
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.orange.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.orange.withOpacity(0.3)),
                ),
                child: const Text(
                  '⚠️ Educational purposes only. NOT for clinical diagnosis.',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 11, color: Colors.orange),
                ),
              ),
              const SizedBox(height: 32),

              // Upload Area
              if (_isAnalyzing) _buildAnalyzingState() else _buildUploadArea(),

              const SizedBox(height: 24),

              // Image Preview
              if (_selectedImage != null && !_isAnalyzing) _buildPreview(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildUploadArea() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 48, horizontal: 24),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.03),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
      ),
      child: Column(
        children: [
          const Icon(Icons.medical_services_outlined, size: 64, color: Color(0xFF4DD0E1)),
          const SizedBox(height: 16),
          const Text(
            'Upload Chest X-Ray',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 8),
          Text(
            'Take a photo or select from gallery',
            style: TextStyle(color: Colors.grey[500], fontSize: 13),
          ),
          const SizedBox(height: 32),

          // Camera button
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: () => _pickImage(ImageSource.camera),
              icon: const Icon(Icons.camera_alt),
              label: const Text('Take Photo'),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF4DD0E1),
                foregroundColor: Colors.black,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
              ),
            ),
          ),
          const SizedBox(height: 12),

          // Gallery button
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: () => _pickImage(ImageSource.gallery),
              icon: const Icon(Icons.photo_library),
              label: const Text('Choose from Gallery'),
              style: OutlinedButton.styleFrom(
                foregroundColor: const Color(0xFF4DD0E1),
                side: const BorderSide(color: Color(0xFF4DD0E1), width: 1.5),
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
              ),
            ),
          ),
          const SizedBox(height: 16),
          Text(
            'Supports: PNG, JPG, DICOM (.dcm)',
            style: TextStyle(color: Colors.grey[600], fontSize: 11),
          ),
        ],
      ),
    );
  }

  Widget _buildAnalyzingState() {
    return AnimatedBuilder(
      animation: _pulseController,
      builder: (context, child) {
        return Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(vertical: 64, horizontal: 24),
          decoration: BoxDecoration(
            color: Color.lerp(
              Colors.cyan.withOpacity(0.02),
              Colors.cyan.withOpacity(0.08),
              _pulseController.value,
            ),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: Color.lerp(
                Colors.cyan.withOpacity(0.2),
                Colors.cyan.withOpacity(0.5),
                _pulseController.value,
              )!,
            ),
          ),
          child: Column(
            children: [
              SizedBox(
                width: 60,
                height: 60,
                child: CircularProgressIndicator(
                  strokeWidth: 3,
                  color: const Color(0xFF4DD0E1).withOpacity(
                    0.5 + _pulseController.value * 0.5,
                  ),
                ),
              ),
              const SizedBox(height: 24),
              const Text(
                '🧠 AI is analyzing your X-ray...',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
              ),
              const SizedBox(height: 8),
              Text(
                'Running deep learning model with Grad-CAM',
                style: TextStyle(color: Colors.grey[500], fontSize: 12),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildPreview() {
    return Column(
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(12),
          child: Image.file(
            _selectedImage!,
            height: 200,
            fit: BoxFit.cover,
          ),
        ),
        const SizedBox(height: 12),
        TextButton.icon(
          onPressed: () => setState(() => _selectedImage = null),
          icon: const Icon(Icons.close, size: 18),
          label: const Text('Remove'),
          style: TextButton.styleFrom(foregroundColor: Colors.red[300]),
        ),
      ],
    );
  }
}
