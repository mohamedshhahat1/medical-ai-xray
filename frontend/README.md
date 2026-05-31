# 📱 Frontend — Medical AI X-Ray

## Options

### 1. Flutter App (Mobile + Web)
```bash
cd flutter_app
flutter run
```

### 2. Web Dashboard
The FastAPI backend serves API docs at `/docs` (Swagger UI).

### API Integration

```dart
// Flutter example
final response = await http.post(
  Uri.parse('http://your-server:8000/predict'),
  body: {'file': imageFile},
);
final result = jsonDecode(response.body);
print(result['prediction']);  // "Pneumonia"
print(result['confidence']);  // 0.93
```

```javascript
// JavaScript example
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
});
const result = await response.json();
```
