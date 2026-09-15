from model.predict import predict_image

image_path = r"D:\SignalScope\data\external_test\dolphin.png"

result = predict_image(image_path)

print("\n--- SignalScope External Test ---")
print(f"Image: {image_path}")
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2f}%")
print(f"AI probability: {result['ai_probability']:.2f}%")
print(f"Real probability: {result['real_probability']:.2f}%")