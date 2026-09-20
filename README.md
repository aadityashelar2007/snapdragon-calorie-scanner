# SnapAI: On-Device Calorie & Meal Scanner 🥗

A privacy-first, on-device meal recognition and nutritional logging application built for **Snapdragon®-powered HP PCs** using **Qualcomm AI Hub** model architectures.

---

## 🌟 Key Features

* **100% On-Device Privacy:** Processes webcam captures and meal photos locally—zero image data sent to cloud APIs.
* **Sub-250ms Local Latency:** Leverages quantized vision backbones optimized for Snapdragon NPUs.
* **Curated Indian & Global Nutrition DB:** Detects regional staples (Kala Chana Chaat, Poha, Idli, Dal Tadka) alongside dry fruits and standard foods.
* **Smart Tableware Fallback:** Intelligently maps container/thali visual tokens to regional meal entries when standard ImageNet models misclassify complex bowls.
* **Real-time Macro Breakdown:** Dynamically recalculates calories, protein, carbs, and fat based on user-adjusted portion weight sliders.

---

## 🏗️ Technical Architecture & Stack

* **UI Framework:** Streamlit 1.30+
* **Vision Backbone:** PyTorch MobileNetV2 (ONNX / QNN Runtime fallback)
* **Hardware Target:** Snapdragon® X Elite NPU
* **NPU Profiling & Benchmark:** Qualcomm AI Hub (`qai-hub`)

---

## ⚡ Qualcomm AI Hub NPU Benchmark Proof

* **Target Device:** Snapdragon X Elite CRD (Windows 11)
* **Runtime:** ONNX / QNN
* **Live AI Hub Benchmark Job:** [View Job Results on Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com/jobs/jp1n7v7kg/)

---

## 🚀 Quickstart (Local Setup)

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/aadityashelar2007/snapdragon-calorie-scanner.git](https://github.com/aadityashelar2007/snapdragon-calorie-scanner.git)
   cd snapdragon-calorie-scanner
