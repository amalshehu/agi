#!/bin/bash
# Setup Azure OpenAI Environment Variables for ARC Prize 2025

echo "🚀 Setting up Azure OpenAI environment for ARC Prize 2025..."

# Set Azure OpenAI credentials
export AZURE_API_KEY="your-azure-api-key-here"
export AZURE_API_VERSION="2024-02-15-preview"
export AZURE_DEPLOYMENT_NAME="prod-gpt"
export AZURE_API_BASE="https://vitalview-ai-test.openai.azure.com"

echo "✅ Azure OpenAI environment variables set!"
echo "📋 Configuration:"
echo "   API Base: $AZURE_API_BASE"
echo "   Deployment: $AZURE_DEPLOYMENT_NAME"
echo "   API Version: $AZURE_API_VERSION"
echo "   API Key: [REDACTED]"
echo ""
echo "🎯 To use with ARC solver:"
echo "   source setup_azure.sh"
echo "   python arc_prize_pipeline.py"
echo ""
echo "💡 Note: The solver works without Azure OpenAI (falls back to template-based hypotheses)"
