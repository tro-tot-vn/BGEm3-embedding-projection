#!/usr/bin/env python3
"""
Upload fixed config.json to HF Hub
(with auto_map for trust_remote_code compatibility)
"""

from huggingface_hub import HfApi

print("=" * 80)
print("📤 Uploading Fixed Config to HF Hub")
print("=" * 80)

api = HfApi()

repo_id = "lamdx4/bge-m3-vietnamese-rental-projection"
config_path = "hf_upload/config.json"

print(f"\n📦 Repository: {repo_id}")
print(f"📄 File: {config_path}")

try:
    print("\n🔄 Uploading config.json...")
    
    api.upload_file(
        path_or_fileobj=config_path,
        path_in_repo="config.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Fix: Add auto_map to config.json for trust_remote_code compatibility"
    )
    
    print("✅ Config uploaded successfully!")
    print(f"\n🌐 View at: https://huggingface.co/{repo_id}/blob/main/config.json")
    
except Exception as e:
    print(f"\n❌ Error uploading: {e}")
    print("\n💡 Make sure you're logged in:")
    print("   huggingface-cli login")

print("\n" + "=" * 80)

