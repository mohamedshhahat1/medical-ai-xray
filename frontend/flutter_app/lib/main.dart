import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'screens/upload_screen.dart';

void main() {
  runApp(const MedicalAIApp());
}

class MedicalAIApp extends StatelessWidget {
  const MedicalAIApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Medical AI X-Ray',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.dark(
          primary: const Color(0xFF4DD0E1),
          secondary: const Color(0xFF7C4DFF),
          surface: const Color(0xFF1A1A2E),
          background: const Color(0xFF0F2027),
          error: const Color(0xFFEF5350),
        ),
        scaffoldBackgroundColor: const Color(0xFF0F2027),
        textTheme: GoogleFonts.interTextTheme(
          ThemeData.dark().textTheme,
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.transparent,
          elevation: 0,
          centerTitle: true,
        ),
      ),
      home: const UploadScreen(),
    );
  }
}
