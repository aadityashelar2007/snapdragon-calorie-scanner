import qai_hub as hub

def run_snapdragon_profiling():
    print("🚀 Connecting to Qualcomm AI Hub Cloud Lab...")
    device = hub.Device("Snapdragon X Elite CRD")
    
    print("📦 Loading MobileNetV2 Vision Backbone for NPU Execution...")
    model_id = "mobilenet_v2"
    
    print(f"⚡ Submitting profiling job to target device: {device.name}...")
    profile_job = hub.submit_profile_job(
        model=model_id,
        device=device,
        options="--target_runtime onnx"
    )
    
    print("\n✅ Profiling Job Submitted Successfully!")
    print(f"🔗 View Live Snapdragon Benchmark Dashboard: {profile_job.url}")

if __name__ == "__main__":
    run_snapdragon_profiling()