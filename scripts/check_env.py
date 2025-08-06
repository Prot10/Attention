#!/usr/bin/env python3
"""Test script to verify .env file setup for wandb."""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv()
    print("✅ Successfully loaded .env file")
except ImportError:
    print("❌ python-dotenv not installed")
    sys.exit(1)

# Check for wandb API key
wandb_key = os.getenv("WANDB_API_KEY")
if wandb_key:
    if wandb_key == "your_wandb_api_key_here":
        print("⚠️  WANDB_API_KEY found in .env but still has placeholder value")
        print("   Please update .env with your actual API key from https://wandb.ai/settings")
    else:
        print(f"✅ WANDB_API_KEY found: {wandb_key[:8]}...{wandb_key[-4:] if len(wandb_key) > 12 else '***'}")
else:
    print("❌ WANDB_API_KEY not found in environment")
    print("   Please add WANDB_API_KEY=your_actual_key to your .env file")

# Check other wandb environment variables
wandb_project = os.getenv("WANDB_PROJECT", "attention-research")
wandb_entity = os.getenv("WANDB_ENTITY")
wandb_mode = os.getenv("WANDB_MODE", "online")

print(f"📊 WANDB_PROJECT: {wandb_project}")
if wandb_entity:
    print(f"👤 WANDB_ENTITY: {wandb_entity}")
print(f"🌐 WANDB_MODE: {wandb_mode}")

# Test wandb import
try:
    import wandb  # noqa: F401

    print("✅ wandb package is available")

    if wandb_key and wandb_key != "your_wandb_api_key_here":
        print("\n🚀 You're ready to start training with wandb tracking!")
        print("   Run: uv run python scripts/test_training.py --framework torch")
    else:
        print("\n⏳ Set up your API key in .env to enable wandb tracking")

except ImportError:
    print("❌ wandb package not installed")
    print("   Run: uv add wandb")
