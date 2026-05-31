import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:http/http.dart' as http;

/// API Service — Handles all communication with the Medical AI backend.
///
/// Connects to the FastAPI server for:
/// - X-ray prediction (diagnosis + confidence)
/// - Grad-CAM heatmap generation
/// - PDF report download
/// - DICOM metadata extraction
class ApiService {
  // Change this to your deployed server URL
  static const String baseUrl = 'http://10.0.2.2:8000'; // Android emulator → localhost
  // static const String baseUrl = 'http://localhost:8000'; // iOS simulator
  // static const String baseUrl = 'https://your-deployed-server.com'; // Production

  /// Predict diagnosis from an X-ray image file.
  ///
  /// Returns a map with:
  /// - prediction: String (e.g., "Pneumonia")
  /// - confidence: double (0.0 - 1.0)
  /// - probabilities: Map<String, double>
  /// - description: String
  /// - severity: String
  static Future<Map<String, dynamic>> predict(File imageFile) async {
    final uri = Uri.parse('$baseUrl/predict');

    final request = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath('file', imageFile.path));

    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Prediction failed: ${response.statusCode} - ${response.body}');
    }
  }

  /// Get Grad-CAM heatmap overlay image.
  ///
  /// Returns the raw PNG bytes of the heatmap overlay.
  static Future<Uint8List> getGradCAM(File imageFile) async {
    final uri = Uri.parse('$baseUrl/predict/gradcam');

    final request = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath('file', imageFile.path));

    final streamedResponse = await request.send();

    if (streamedResponse.statusCode == 200) {
      return await streamedResponse.stream.toBytes();
    } else {
      throw Exception('Grad-CAM generation failed: ${streamedResponse.statusCode}');
    }
  }

  /// Generate and download PDF report.
  ///
  /// Returns the raw PDF bytes.
  static Future<Uint8List> getReport(File imageFile, {String? patientName, String? patientId}) async {
    final uri = Uri.parse('$baseUrl/report');

    final request = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath('file', imageFile.path));

    if (patientName != null) request.fields['patient_name'] = patientName;
    if (patientId != null) request.fields['patient_id'] = patientId;

    final streamedResponse = await request.send();

    if (streamedResponse.statusCode == 200) {
      return await streamedResponse.stream.toBytes();
    } else {
      throw Exception('Report generation failed: ${streamedResponse.statusCode}');
    }
  }

  /// Check server health status.
  static Future<Map<String, dynamic>> healthCheck() async {
    final uri = Uri.parse('$baseUrl/health');
    final response = await http.get(uri);

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Server unreachable');
    }
  }

  /// Get DICOM metadata from a .dcm file.
  static Future<Map<String, dynamic>> getDicomMetadata(File dicomFile) async {
    final uri = Uri.parse('$baseUrl/dicom/metadata');

    final request = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath('file', dicomFile.path));

    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('DICOM metadata extraction failed');
    }
  }
}
