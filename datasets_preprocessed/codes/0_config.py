#!/usr/bin/env python3
"""
Centralized Configuration for MMS Benchmark Pipeline

This file contains all API configurations and settings used across the pipeline.
All scripts import from this file to ensure consistency.
"""

import os

# ==================== Azure OpenAI Configuration ====================
# Azure OpenAI Endpoint and Keys
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"

# Azure OpenAI Deployment Names
AZURE_DEPLOYMENT_GPT4O_MINI = "gpt-5.2"  # For frame descriptions and text generation
AZURE_DEPLOYMENT_GPT4O = "gpt-5.2"  # For document generation

# ==================== Default Pipeline Settings ====================
# Default GPT model for document generation (Step 3)
DEFAULT_GPT_MODEL = AZURE_DEPLOYMENT_GPT4O

# Default similarity threshold for merging shots (Step 3)
DEFAULT_SIMILARITY_THRESHOLD = 1.0  # Range: 0.0 to 1.0 (higher = less merging)

# Maximum frames per API request (Step 2)
MAX_FRAMES_PER_REQUEST = 10

# ==================== Helper Functions ====================
def create_azure_openai_client():
    """
    Create and return an Azure OpenAI client instance

    Returns:
        AzureOpenAI: Configured Azure OpenAI client
    """
    from openai import AzureOpenAI

    return AzureOpenAI(
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
    )


def get_deployment_name(model_name=None):
    """
    Get Azure deployment name based on model preference

    Args:
        model_name: Optional model name or deployment name

    Returns:
        str: Azure deployment name
    """
    if model_name is None:
        return DEFAULT_GPT_MODEL

    # Map common model names to Azure deployments
    model_map = {
        "gpt-4o-mini": AZURE_DEPLOYMENT_GPT4O_MINI,
        "gpt-4o-mini-risk": AZURE_DEPLOYMENT_GPT4O_MINI,
        "gpt-4o": AZURE_DEPLOYMENT_GPT4O,
        "gpt-4o-risk": AZURE_DEPLOYMENT_GPT4O,
    }

    return model_map.get(model_name, model_name)


# ==================== Configuration Summary ====================
if __name__ == "__main__":
    print("="*80)
    print("MMS Benchmark Pipeline Configuration")
    print("="*80)
    print(f"Azure Endpoint:        {AZURE_OPENAI_ENDPOINT}")
    print(f"API Version:           {AZURE_OPENAI_API_VERSION}")
    print(f"GPT-4o Mini Deployment: {AZURE_DEPLOYMENT_GPT4O_MINI}")
    print(f"GPT-4o Deployment:     {AZURE_DEPLOYMENT_GPT4O}")
    print(f"Default Model:         {DEFAULT_GPT_MODEL}")
    print(f"Default Threshold:     {DEFAULT_SIMILARITY_THRESHOLD}")
    print(f"Max Frames/Request:    {MAX_FRAMES_PER_REQUEST}")
    print("="*80)
